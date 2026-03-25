#include <opencv2/dnn/all_layers.hpp>
//#include <opencv2/core/opencl/ocl_defs.hpp>
#include <cvconfig.h>

//#include <layers/layers_common.hpp>
#include <op_cuda.hpp>
#include <op_halide.hpp>
#include <op_inf_engine.hpp>
#include <ie_ngraph.hpp>
#include <op_webnn.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/trace.hpp>

#ifdef HAVE_CUDA
//#include <cuda4dnn/primitives/scale_shift.hpp>
#include "axpy.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
	namespace dnn
	{

		class AxpyFastLayerImpl CV_FINAL : public Layer
		{
		public:
#ifdef HAVE_WEBNN
			mutable int dims;
			mutable int numChannels;
#endif
			AxpyFastLayerImpl(const LayerParams& params)
			{
				setParamsFrom(params);
			}

			bool getMemoryShapes(const std::vector<MatShape>& inputs,
				const int requiredOutputs,
				std::vector<MatShape>& outputs,
				std::vector<MatShape>& internals) const CV_OVERRIDE
			{
				outputs.assign(1, inputs[1]);
#ifdef HAVE_WEBNN
				dims = inputs[0].size();
				numChannels = 1;
				if (inputs.size() > 1)
				{
					for (const size_t& dim : inputs[1])
						numChannels *= dim;
				}
#endif
				return true;
			}

			virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
			{
				std::vector<Mat> inputs;
				inputs_arr.getMatVector(inputs);
				CV_Assert(inputs.size() == 3);
			}

			virtual bool supportBackend(int backendId) CV_OVERRIDE
			{
#ifdef HAVE_INF_ENGINE
				if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
					return true;
#endif
				return backendId == DNN_BACKEND_OPENCV ||
					backendId == DNN_BACKEND_CUDA ||
					backendId == DNN_BACKEND_HALIDE ||
					backendId == DNN_BACKEND_WEBNN;
			}

			void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
			{
				CV_TRACE_FUNCTION();
				CV_TRACE_ARG_VALUE(name, "name", name.c_str());

				if (inputs_arr.depth() == CV_16F)
				{
					forward_fallback(inputs_arr, outputs_arr, internals_arr);
					return;
				}

				std::vector<Mat> inputs, outputs;
				inputs_arr.getMatVector(inputs);
				outputs_arr.getMatVector(outputs);

				CV_Assert_N(outputs.size() == 1, inputs.size() == 3);

				Mat& inpBlob = inputs[1];
				Mat& outBlob = outputs[0];
				// There is a mode when we multiply a first blob by a second one
				// instead of trainable weights.
				Mat weights = inputs[0].reshape(1, 1);
				Mat bias = inputs[2].reshape(1, 1);

				MatShape inpShape0 = shape(inputs[0]);
				MatShape inpShape1 = shape(inputs[1]);
				MatShape inpShape2 = shape(inputs[2]);

				// TODO: 向こうが想定しているbiasがこちらが想定しているshapeと違う想定っぽいので計算処理を書き直す
				// こちらが想定しているの: weights.shape == bias.shape
				// 向こうが想定しているの: inpBlob.shape == bias.shape

				MatShape inpShape = shape(inpBlob);
				const int numWeights = weights.total();
				CV_Assert(numWeights != 0);
				CV_CheckEQ(weights.total(), bias.total(), "Incompatible weights/bias blobs");

				if (weights.total() == 1)
				{
					// The total() of bias should be same as weights.
					inpBlob.convertTo(outBlob, CV_32F, weights.at<float>(0), bias.at<float>(0));
					return;
				}

				int endAxis;
				for (endAxis = 1; endAxis <= inpBlob.dims; ++endAxis)
				{
					if (total(inpShape, 0, endAxis) == numWeights)
						break;
				}
				CV_Assert(total(inpShape, 0, endAxis) == numWeights);
				CV_Assert(numWeights == bias.total());
				CV_CheckTypeEQ(inpBlob.type(), CV_32FC1, ""); CV_CheckTypeEQ(outBlob.type(), CV_32FC1, "");

				int numSlices = total(inpShape, 0, 0);
				float* inpData = (float*)inpBlob.data;
				float* outData = (float*)outBlob.data;

				if (endAxis != inpBlob.dims)
				{
					float* weightsData = (float*)weights.data;
					float* biasesData = (float*)bias.data;
					int spatialSize = total(inpShape, endAxis);  // spatialSize != 1
					for (int i = 0; i < numSlices; ++i)
					{
						for (int j = 0; j < numWeights; ++j)
						{
							float w = weightsData ? weightsData[j] : 1;
							float b = biasesData ? biasesData[j] : 0;
							Mat inpSlice(1, spatialSize, CV_32F, inpData);
							Mat outSlice(1, spatialSize, CV_32F, outData);

							inpSlice.convertTo(outSlice, CV_32F, w, b);

							inpData += spatialSize;
							outData += spatialSize;
						}
					}
				}
				else
				{
					for (int i = 0; i < numSlices; ++i)
					{
						Mat inpSlice(1, numWeights, CV_32F, inpData);
						Mat outSlice(1, numWeights, CV_32F, outData);

						multiply(inpSlice, weights, outSlice);
						add(outSlice, bias, outSlice);

						inpData += numWeights;
						outData += numWeights;
					}
				}
			}

#ifdef HAVE_CUDA
			Ptr<BackendNode> initCUDA(
				void* context_,
				const std::vector<Ptr<BackendWrapper>>& inputs,
				const std::vector<Ptr<BackendWrapper>>& outputs
			) override
			{
				auto context = reinterpret_cast<csl::CSLContext*>(context_);

				CV_Assert(inputs.size() == 3);

				return make_cuda_node<cuda4dnn::AxpyOp>(preferableTarget, std::move(context->stream));
			}
#endif

			virtual Ptr<BackendNode> tryAttach(const Ptr<BackendNode>& node) CV_OVERRIDE
			{
				switch (node->backendId)
				{
				case DNN_BACKEND_HALIDE:
				{
#ifdef HAVE_HALIDE
					auto base = node.dynamicCast<HalideBackendNode>();
					Halide::Func& input = base->funcs.back();
					Halide::Var x("x"), y("y"), c("c"), n("n");
					Halide::Func top = attachHalide(input(x, y, c, n));
					return Ptr<BackendNode>(new HalideBackendNode(base, top));
#endif  // HAVE_HALIDE
					break;
				}
				}
				return Ptr<BackendNode>();
			}

			virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> >& inputs) CV_OVERRIDE
			{
#ifdef HAVE_HALIDE
				Halide::Buffer<float> input = halideBuffer(inputs[0]);
				Halide::Var x("x"), y("y"), c("c"), n("n");
				Halide::Func top = attachHalide(input(x, y, c, n));
				return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
				return Ptr<BackendNode>();
			}

#ifdef HAVE_HALIDE
			// attachHalide can work both with Halide::Buffer and Halide::Func. In the
			// second case it will be a fusion.
			Halide::Func attachHalide(const Halide::Expr& input)
			{
				Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
				Halide::Var x("x"), y("y"), c("c"), n("n");

				const int numChannels = blobs[0].total();

				Halide::Expr topExpr = input;
				if (hasWeights)
				{
					auto weights = wrapToHalideBuffer(blobs[0], { numChannels });
					topExpr *= weights(c);
				}
				if (hasBias)
				{
					auto bias = wrapToHalideBuffer(blobs.back(), { numChannels });
					topExpr += bias(c);
				}
				top(x, y, c, n) = topExpr;
				return top;
			}
#endif  // HAVE_HALIDE


#ifdef HAVE_DNN_NGRAPH
			virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
			{
				auto ieInpNode0 = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
				ov::Output<ov::Node> ieInpNode1;
				if (nodes.size() > 1)
					ieInpNode1 = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;

				size_t numChannels = 1;
				if (blobs.empty())
					for (const size_t& dim : ieInpNode1.get_shape())
						numChannels *= dim;
				else
					numChannels = blobs[0].total();

				std::vector<size_t> shape(ieInpNode0.get_shape().size(), 1);
				int cAxis = normalize_axis(axis, shape.size());
				shape[cAxis] = numChannels;

				std::shared_ptr<ov::Node> node;
				if (hasWeights)
				{
					ov::Output<ov::Node> weight = blobs.empty() ? ieInpNode1 :
						std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape(shape), blobs[0].data);
					node = std::make_shared<ov::op::v1::Multiply>(ieInpNode0, weight, ov::op::AutoBroadcastType::NUMPY);
				}
				if (hasBias || !hasWeights)
				{
					ov::Output<ov::Node> bias;
					if (hasBias)
					{
						bias = blobs.empty() ? ieInpNode1 :
							std::make_shared<ov::op::v0::Constant>(ov::element::f32,
								ov::Shape(shape), blobs.back().data);
					}
					else
						bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
							ov::Shape(shape), std::vector<float>(numChannels, 0).data());
					node = std::make_shared<ov::op::v1::Add>(node, bias, ov::op::AutoBroadcastType::NUMPY);
				}
				return Ptr<BackendNode>(new InfEngineNgraphNode(node));
			}
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
			virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
			{
				Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
				auto& webnnInpOperand0 = node->operand;
				auto& webnnGraphBuilder = node->net->builder;
				auto webnnInpOperand1 = nodes.size() > 1 ? nodes[1].dynamicCast<WebnnBackendNode>()->operand : nullptr;
				auto webnnInpOperand2 = nodes.size() > 2 ? nodes[1].dynamicCast<WebnnBackendNode>()->operand : nullptr;
				std::vector<int32_t> shape(dims, 1);

				size_t channels = 1;
				if (blobs.empty())
					channels = numChannels;
				else
					channels = blobs[0].total();

				int cAxis = normalize_axis(axis, shape.size());
				shape[cAxis] = channels;

				ml::Operand operand = webnnInpOperand0;
				if (hasWeights)
				{
					ml::Operand webnnWeights = blobs.empty() ? webnnInpOperand1 : webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(blobs[0]), blobs[0].data, blobs[0].total() * blobs[0].elemSize(), ml::OperandType::Float32);
					webnnWeights = webnnGraphBuilder.Reshape(webnnWeights, shape.data(), shape.size());
					operand = webnnGraphBuilder.Mul(operand, webnnWeights);
				}
				if (hasBias)
				{
					ml::Operand webnnBias;
					if (!hasWeights)
						webnnBias = blobs.empty() ? webnnInpOperand1 : webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(blobs.back()), blobs.back().data, blobs.back().total() * blobs.back().elemSize(), ml::OperandType::Float32);
					else
						webnnBias = blobs.empty() ? webnnInpOperand2 : webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(blobs.back()), blobs.back().data, blobs.back().total() * blobs.back().elemSize(), ml::OperandType::Float32);
					webnnBias = webnnGraphBuilder.Reshape(webnnBias, shape.data(), shape.size());
					operand = webnnGraphBuilder.Add(operand, webnnBias);
				}

				return Ptr<BackendNode>(new WebnnBackendNode(operand));
			}
#endif


			void getScaleShift(Mat& scale, Mat& shift) const CV_OVERRIDE
			{
				scale = Mat();
				shift = Mat();
			}

			//bool tryQuantize(const std::vector<std::vector<float> >& scales,
			//	const std::vector<std::vector<int> >& zeropoints, LayerParams& params) CV_OVERRIDE
			//{
			//	params.set("input_scales", DictValue::arrayReal(scales[0].data(), scales[0].size()));
			//	params.set("input_zeropoints", DictValue::arrayInt(zeropoints[0].data(), zeropoints[0].size()));
			//	return true;
			//}

			virtual int64 getFLOPS(const std::vector<MatShape>& inputs,
				const std::vector<MatShape>& outputs) const CV_OVERRIDE
			{
				CV_UNUSED(outputs); // suppress unused variable warning
				long flops = 0;
				for (int i = 0; i < inputs.size(); i++)
				{
					flops += 3 * total(inputs[i]);
				}
				return flops;
			}

			static Ptr<AxpyFastLayerImpl> create(const LayerParams& params)
			{
				return Ptr<AxpyFastLayerImpl>(new AxpyFastLayerImpl(params));
			}
		};

	}  // namespace dnn
}  // namespace cv

# include <opencv2/dnn/layer.details.hpp>

void reg2()
{
	CV_DNN_REGISTER_LAYER_CLASS(AxpyFast, cv::dnn::AxpyFastLayerImpl);
}
