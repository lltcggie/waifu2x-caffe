#include "cNet.h"
#include <caffe/caffe.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <rapidjson/document.h>
#include <opencv2/imgproc.hpp>

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.


template<typename BufType>
static bool readFile(boost::iostreams::stream<boost::iostreams::file_descriptor_source> &is, std::vector<BufType> &buf)
{
	if (!is)
		return false;

	const auto size = is.seekg(0, std::ios::end).tellg();
	is.seekg(0, std::ios::beg);

	buf.resize((size / sizeof(BufType)) + (size % sizeof(BufType)));
	is.read(buf.data(), size);
	if (is.gcount() != size)
		return false;

	return true;
}

template<typename BufType>
static bool readFile(const boost::filesystem::path &path, std::vector<BufType> &buf)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor_source> is;

	try
	{
		is.open(path, std::ios_base::in | std::ios_base::binary);
	}
	catch (...)
	{
		return false;
	}

	return readFile(is, buf);
}

static Waifu2x::eWaifu2xError readProtoText(const boost::filesystem::path &path, ::google::protobuf::Message* proto)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor_source> is;

	try
	{
		is.open(path, std::ios_base::in);
	}
	catch (...)
	{
		return Waifu2x::eWaifu2xError_FailedOpenModelFile;
	}

	if (!is)
		return Waifu2x::eWaifu2xError_FailedOpenModelFile;

	std::vector<char> tmp;
	if (!readFile(is, tmp))
		return Waifu2x::eWaifu2xError_FailedParseModelFile;

	google::protobuf::io::ArrayInputStream input(tmp.data(), tmp.size());
	const bool success = google::protobuf::TextFormat::Parse(&input, proto);

	if (!success)
		return Waifu2x::eWaifu2xError_FailedParseModelFile;

	return Waifu2x::eWaifu2xError_OK;
}

static Waifu2x::eWaifu2xError writeProtoBinary(const ::google::protobuf::Message& proto, const boost::filesystem::path &path)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor> os;

	try
	{
		os.open(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
	}
	catch (...)
	{
		return Waifu2x::eWaifu2xError_FailedOpenModelFile;
	}

	if (!os)
		return Waifu2x::eWaifu2xError_FailedWriteModelFile;

	if (!proto.SerializePartialToOstream(&os))
		return Waifu2x::eWaifu2xError_FailedWriteModelFile;

	return Waifu2x::eWaifu2xError_OK;
}

static Waifu2x::eWaifu2xError readProtoBinary(const boost::filesystem::path &path, ::google::protobuf::Message* proto)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor_source> is;

	try
	{
		is.open(path, std::ios_base::in | std::ios_base::binary);
	}
	catch (...)
	{
		return Waifu2x::eWaifu2xError_FailedOpenModelFile;
	}

	if (!is)
		return Waifu2x::eWaifu2xError_FailedOpenModelFile;

	std::vector<char> tmp;
	if (!readFile(is, tmp))
		return Waifu2x::eWaifu2xError_FailedParseModelFile;

	google::protobuf::io::ArrayInputStream input(tmp.data(), tmp.size());

	google::protobuf::io::CodedInputStream coded_input(&input);
	coded_input.SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

	const bool success = proto->ParseFromCodedStream(&coded_input);
	if (!success)
		return Waifu2x::eWaifu2xError_FailedParseModelFile;

	return Waifu2x::eWaifu2xError_OK;
}

namespace
{
	Waifu2x::eWaifu2xError ReadJson(const boost::filesystem::path &info_path, rapidjson::Document &d, std::vector<char> &jsonBuf)
	{
		try
		{
			boost::iostreams::stream<boost::iostreams::file_descriptor_source> is;

			try
			{
				is.open(info_path, std::ios_base::in | std::ios_base::binary);
			}
			catch (...)
			{
				return Waifu2x::eWaifu2xError_FailedOpenModelFile;
			}

			if (!is)
				return Waifu2x::eWaifu2xError_FailedOpenModelFile;

			const size_t size = is.seekg(0, std::ios::end).tellg();
			is.seekg(0, std::ios::beg);

			jsonBuf.resize(size + 1);
			is.read(jsonBuf.data(), jsonBuf.size());

			jsonBuf[jsonBuf.size() - 1] = '\0';

			d.Parse(jsonBuf.data());
		}
		catch (...)
		{
			return Waifu2x::eWaifu2xError_FailedParseModelFile;
		}

		return Waifu2x::eWaifu2xError_OK;
	}
};


cNet::cNet() : mModelScale(0), mInnerScale(0), mNetOffset(0), mInputPlane(0), mHasNoiseScaleModel(false)
{}

cNet::~cNet()
{}

Waifu2x::eWaifu2xError cNet::GetInfo(const boost::filesystem::path & info_path, stInfo &info)
{
	rapidjson::Document d;
	std::vector<char> jsonBuf;

	try
	{
		Waifu2x::eWaifu2xError ret;

		ret = ReadJson(info_path, d, jsonBuf);
		if (ret != Waifu2x::eWaifu2xError_OK)
			return ret;

		const auto name = d["name"].GetString();
		const auto arch_name = d["arch_name"].GetString();
		const bool has_noise_scale = d.HasMember("has_noise_scale") && d["has_noise_scale"].GetBool() ? true : false;
		const int channels = d["channels"].GetInt();

		info.name = name;
		info.arch_name = arch_name;
		info.has_noise_scale = has_noise_scale;
		info.channels = channels;

		if (d.HasMember("offset"))
		{
			const int offset = d["offset"].GetInt();

			info.noise.offset = offset;
			info.scale.offset = offset;
			info.noise_scale.offset = offset;
		}

		if (d.HasMember("scale_factor"))
		{
			const int scale_factor = d["scale_factor"].GetInt();

			info.noise.scale_factor = scale_factor;
			info.scale.scale_factor = scale_factor;
			info.noise_scale.scale_factor = scale_factor;
		}

		if (d.HasMember("offset_noise"))
		{
			const int offset = d["offset_noise"].GetInt();
			info.noise.offset = offset;
		}

		if (d.HasMember("scale_factor_noise"))
		{
			const int scale_factor = d["scale_factor_noise"].GetInt();
			info.noise.scale_factor = scale_factor;
		}

		if (d.HasMember("offset_scale"))
		{
			const int offset = d["offset_scale"].GetInt();
			info.scale.offset = offset;
		}

		if (d.HasMember("scale_factor_scale"))
		{
			const int scale_factor = d["scale_factor_scale"].GetInt();
			info.scale.scale_factor = scale_factor;
		}

		if (d.HasMember("offset_noise_scale"))
		{
			const int offset = d["offset_noise_scale"].GetInt();
			info.noise_scale.offset = offset;
		}

		if (d.HasMember("scale_factor_noise_scale"))
		{
			const int scale_factor = d["scale_factor_noise_scale"].GetInt();
			info.noise_scale.scale_factor = scale_factor;
		}
	}
	catch (...)
	{
		return Waifu2x::eWaifu2xError_FailedParseModelFile;
	}

	return Waifu2x::eWaifu2xError_OK;
}

// モデルファイルからネットワークを構築
// processでcudnnが指定されなかった場合はcuDNNが呼び出されないように変更する
Waifu2x::eWaifu2xError cNet::ConstractNet(const Waifu2x::eWaifu2xModelType mode, const boost::filesystem::path &model_path, const boost::filesystem::path &param_path, const stInfo &info, const std::string &process)
{
	Waifu2x::eWaifu2xError ret;

	mMode = mode;

	LoadParamFromInfo(mode, info);

	boost::filesystem::path modelbin_path = model_path;
	modelbin_path += ".protobin";
	boost::filesystem::path caffemodel_path = param_path;
	caffemodel_path += ".caffemodel";

	caffe::NetParameter param_model;
	caffe::NetParameter param_caffemodel;

	const auto retModelBin = readProtoBinary(modelbin_path, &param_model);
	const auto retParamBin = readProtoBinary(caffemodel_path, &param_caffemodel);

	if (retModelBin == Waifu2x::eWaifu2xError_OK && retParamBin == Waifu2x::eWaifu2xError_OK)
	{
		ret = SetParameter(param_model, process);
		if (ret != Waifu2x::eWaifu2xError_OK)
			return ret;

		if (!caffe::UpgradeNetAsNeeded(caffemodel_path.string(), &param_caffemodel))
			return Waifu2x::eWaifu2xError_FailedParseModelFile;

		mNet = boost::shared_ptr<caffe::Net<float>>(new caffe::Net<float>(param_model));
		mNet->CopyTrainedLayersFrom(param_caffemodel);
	}
	else
	{
		const auto ret = LoadParameterFromJson(model_path, param_path, modelbin_path, caffemodel_path, process);
		if (ret != Waifu2x::eWaifu2xError_OK)
			return ret;
	}

	const auto &inputs = mNet->input_blobs();
	if (inputs.empty())
		return Waifu2x::eWaifu2xError_FailedConstructModel;

	if (mInputPlane != inputs[0]->channels())
		return Waifu2x::eWaifu2xError_FailedConstructModel;

	return Waifu2x::eWaifu2xError_OK;
}

void cNet::LoadParamFromInfo(const Waifu2x::eWaifu2xModelType mode, const stInfo &info)
{
	mModelScale = 2; // TODO: 動的に設定するようにする

	stInfo::stParam param;

	switch (mode)
	{
	case Waifu2x::eWaifu2xModelTypeNoise:
		param = info.noise;
		break;

	case Waifu2x::eWaifu2xModelTypeScale:
		param = info.scale;
		break;

	case Waifu2x::eWaifu2xModelTypeNoiseScale:
		param = info.noise_scale;
		break;

	case Waifu2x::eWaifu2xModelTypeAutoScale:
		param = info.noise_scale;
		break;
	}

	mInnerScale = param.scale_factor;
	mNetOffset = param.offset;
	mInputPlane = info.channels;
	mHasNoiseScaleModel = info.has_noise_scale;
}

Waifu2x::eWaifu2xError cNet::SetParameter(caffe::NetParameter &param, const std::string &process) const
{
	param.mutable_state()->set_phase(caffe::TEST);

	{
		auto input_layer = param.mutable_layer(0);
		auto mid = input_layer->mutable_input_param()->mutable_shape();
		if (mid->size() != 1 || mid->Mutable(0)->dim_size() != 4)
			return Waifu2x::eWaifu2xError_FailedParseModelFile;
		mid->Mutable(0)->set_dim(0, 1);
		mid->Mutable(0)->set_dim(2, 142);
		mid->Mutable(0)->set_dim(3, 142);
	}

	for (int i = 0; i < param.layer_size(); i++)
	{
		caffe::LayerParameter *layer_param = param.mutable_layer(i);
		const std::string& type = layer_param->type();
		if (type == "Convolution")
		{
			if (process == "cudnn")
				layer_param->mutable_convolution_param()->set_engine(caffe::ConvolutionParameter_Engine_CUDNN);
			else
				layer_param->mutable_convolution_param()->set_engine(caffe::ConvolutionParameter_Engine_CAFFE);
		}
		else if (type == "ReLU")
		{
			if (process == "cudnn")
				layer_param->mutable_relu_param()->set_engine(caffe::ReLUParameter_Engine_CUDNN);
			else
				layer_param->mutable_relu_param()->set_engine(caffe::ReLUParameter_Engine_CAFFE);
		}
	}

	return Waifu2x::eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError cNet::LoadParameterFromJson(const boost::filesystem::path &model_path, const boost::filesystem::path &param_path
	, const boost::filesystem::path &modelbin_path, const boost::filesystem::path &caffemodel_path, const std::string &process)
{
	Waifu2x::eWaifu2xError ret;

	caffe::NetParameter param;
	ret = readProtoText(model_path, &param);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	ret = writeProtoBinary(param, modelbin_path);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	ret = SetParameter(param, process);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	mNet = boost::shared_ptr<caffe::Net<float>>(new caffe::Net<float>(param));

	rapidjson::Document d;
	std::vector<char> jsonBuf;

	try
	{
		boost::iostreams::stream<boost::iostreams::file_descriptor_source> is;

		try
		{
			is.open(param_path, std::ios_base::in | std::ios_base::binary);
		}
		catch (...)
		{
			return Waifu2x::eWaifu2xError_FailedOpenModelFile;
		}

		if (!is)
			return Waifu2x::eWaifu2xError_FailedOpenModelFile;

		const size_t size = is.seekg(0, std::ios::end).tellg();
		is.seekg(0, std::ios::beg);

		jsonBuf.resize(size + 1);
		is.read(jsonBuf.data(), jsonBuf.size());

		jsonBuf[jsonBuf.size() - 1] = '\0';

		d.Parse(jsonBuf.data());
	}
	catch (...)
	{
		return Waifu2x::eWaifu2xError_FailedParseModelFile;
	}

	if (d.Size() != 7)
		return Waifu2x::eWaifu2xError_FailedParseModelFile;

	int inputPlane = 0;
	int outputPlane = 0;
	try
	{
		inputPlane = d[0]["nInputPlane"].GetInt();
		outputPlane = d[d.Size() - 1]["nOutputPlane"].GetInt();
	}
	catch (...)
	{
		return Waifu2x::eWaifu2xError_FailedParseModelFile;
	}

	if (inputPlane == 0 || outputPlane == 0)
		return Waifu2x::eWaifu2xError_FailedParseModelFile;

	if (inputPlane != outputPlane)
		return Waifu2x::eWaifu2xError_FailedParseModelFile;

	//if (param.layer_size() < 17)
	//	return Waifu2x::eWaifu2xError_FailedParseModelFile;

	std::vector<boost::shared_ptr<caffe::Layer<float>>> list;
	auto &v = mNet->layers();
	for (auto &l : v)
	{
		auto lk = l->type();
		auto &bv = l->blobs();
		if (bv.size() > 0)
			list.push_back(l);
	}

	try
	{
		std::vector<float> weightList;
		std::vector<float> biasList;

		int count = 0;
		for (auto it = d.Begin(); it != d.End(); ++it)
		{
			const auto &weight = (*it)["weight"];
			const auto nInputPlane = (*it)["nInputPlane"].GetInt();
			const auto nOutputPlane = (*it)["nOutputPlane"].GetInt();
			const auto kW = (*it)["kW"].GetInt();
			const auto &bias = (*it)["bias"];

			auto leyer = list[count];

			auto &b0 = leyer->blobs()[0];
			auto &b1 = leyer->blobs()[1];

			float *b0Ptr = nullptr;
			float *b1Ptr = nullptr;

			if (caffe::Caffe::mode() == caffe::Caffe::CPU)
			{
				b0Ptr = b0->mutable_cpu_data();
				b1Ptr = b1->mutable_cpu_data();
			}
			else
			{
				b0Ptr = b0->mutable_gpu_data();
				b1Ptr = b1->mutable_gpu_data();
			}

			const auto WeightSize1 = weight.Size();
			const auto WeightSize2 = weight[0].Size();
			const auto KernelHeight = weight[0][0].Size();
			const auto KernelWidth = weight[0][0][0].Size();

			if (!(b0->count() == WeightSize1 * WeightSize2 * KernelHeight * KernelWidth))
				return Waifu2x::eWaifu2xError_FailedConstructModel;

			if (!(b1->count() == bias.Size()))
				return Waifu2x::eWaifu2xError_FailedConstructModel;

			weightList.resize(0);
			biasList.resize(0);

			size_t weightCount = 0;
			for (auto it2 = weight.Begin(); it2 != weight.End(); ++it2)
			{
				for (auto it3 = (*it2).Begin(); it3 != (*it2).End(); ++it3)
				{
					for (auto it4 = (*it3).Begin(); it4 != (*it3).End(); ++it4)
					{
						for (auto it5 = (*it4).Begin(); it5 != (*it4).End(); ++it5)
							weightList.push_back((float)it5->GetDouble());
					}
				}
			}

			caffe::caffe_copy(b0->count(), weightList.data(), b0Ptr);

			for (auto it2 = bias.Begin(); it2 != bias.End(); ++it2)
				biasList.push_back((float)it2->GetDouble());

			caffe::caffe_copy(b1->count(), biasList.data(), b1Ptr);

			count++;
		}

		mNet->ToProto(&param);

		ret = writeProtoBinary(param, caffemodel_path);
		if (ret != Waifu2x::eWaifu2xError_OK)
			return ret;
	}
	catch (...)
	{
		return Waifu2x::eWaifu2xError_FailedConstructModel;
	}

	return Waifu2x::eWaifu2xError_OK;
}

int cNet::GetInputPlane() const
{
	return mInputPlane;
}

int cNet::GetInnerScale() const
{
	return mInnerScale;
}

int cNet::GetNetOffset() const
{
	return mNetOffset;
}

int cNet::GetScale() const
{
	return mModelScale;
}

int cNet::GetInputMemorySize(const int crop_w, const int crop_h, const int outer_padding, const int batch_size) const
{
	const int InputPadding = mNetOffset + outer_padding;
	const auto input_block_width = crop_w + InputPadding * 2;
	const auto input_block_height = crop_h + InputPadding * 2;

	const int input_block_plane_size = input_block_width * input_block_height * mInputPlane;

	return input_block_plane_size * batch_size * sizeof(float);
	
}

int cNet::GetOutputMemorySize(const int crop_w, const int crop_h, const int outer_padding, const int batch_size) const
{
	const int InputPadding = mNetOffset + outer_padding;
	const auto input_block_width = crop_w + InputPadding * 2;
	const auto input_block_height = crop_h + InputPadding * 2;

	const auto output_block_width = input_block_width * mInnerScale - mNetOffset * 2;
	const auto output_block_height = input_block_height * mInnerScale - mNetOffset * 2;

	const int output_block_plane_size = output_block_width * output_block_height * mInputPlane;

	return output_block_plane_size * batch_size * sizeof(float);
}

// ネットワークを使って画像を再構築する
Waifu2x::eWaifu2xError cNet::ReconstructImage(const bool UseTTA, const int crop_w, const int crop_h, const int outer_padding, const int batch_size, float *outputBlockBuf, const cv::Mat &inMat, cv::Mat &outMat)
{
	const auto InputHeight = inMat.size().height;
	const auto InputWidth = inMat.size().width;
	const auto InputLine = inMat.step1();

	assert(inMat.channels() == 1 || inMat.channels() == 3);

	const int InputPadding = mNetOffset + outer_padding; // 入力パディング

	const auto NoPaddingInputWidth = InputWidth - InputPadding * 2; // パディングを除いた入力画像サイズ(横)
	const auto NoPaddingInputHeight = InputHeight - InputPadding * 2; // パディングを除いた入力画像サイズ(縦)

	cv::Mat outim(NoPaddingInputHeight * mInnerScale, NoPaddingInputWidth * mInnerScale, inMat.type());

	// float *imptr = (float *)im.data;
	float *imptr = (float *)outim.data;

	const auto input_block_width = crop_w + InputPadding * 2; // 入力ブロックサイズ(横)
	const auto input_block_height = crop_h + InputPadding * 2; // 入力ブロックサイズ(縦)

	const auto output_block_width = input_block_width * mInnerScale - mNetOffset * 2; // 出力ブロックサイズ(横)
	const auto output_block_height = input_block_height * mInnerScale - mNetOffset * 2; // 出力ブロックサイズ(縦)

	const auto output_crop_block_width = crop_w * mInnerScale; // クロップ後の出力ブロックサイズ(横)
	const auto output_crop_block_height = crop_h * mInnerScale; // クロップ後の出力ブロックサイズ(縦)

	const auto output_crop_w = (output_block_width - crop_w * mInnerScale) / 2; // 出力後のクロップサイズ
	const auto output_crop_h = (output_block_height - crop_h * mInnerScale) / 2; // 出力後のクロップサイズ

	assert(NoPaddingInputWidth % crop_w == 0);
	assert(NoPaddingInputHeight % crop_h == 0);

	try
	{
		auto input_blobs = mNet->input_blobs();

		assert(input_blobs.size() > 0);

		auto input_blob = mNet->input_blobs()[0];

		input_blob->Reshape(batch_size, mInputPlane, input_block_height, input_block_width);

		assert(inMat.channels() == mInputPlane);
		assert(input_blob->shape(1) == mInputPlane);

		const int WidthNum = NoPaddingInputWidth / crop_w;
		const int HeightNum = NoPaddingInputHeight / crop_h;

		const int BlockNum = WidthNum * HeightNum;

		const int input_block_plane_size = input_block_width * input_block_height * mInputPlane;
		const int output_block_plane_size = output_block_width * output_block_height * mInputPlane;

		// 画像は(消費メモリの都合上)block_size*block_sizeに分けて再構築する
		for (int num = 0; num < BlockNum; num += batch_size)
		{
			const int processNum = (BlockNum - num) >= batch_size ? batch_size : BlockNum - num;

			if (processNum < batch_size)
				input_blob->Reshape(processNum, mInputPlane, input_block_height, input_block_width);

			for (int n = 0; n < processNum; n++)
			{
				const int wn = (num + n) % WidthNum;
				const int hn = (num + n) / WidthNum;

				const int w = wn * crop_w;
				const int h = hn * crop_h;

				assert(w + input_block_width <= InputWidth && h + input_block_height <= InputHeight);

				cv::Mat someimg = inMat(cv::Rect(w, h, input_block_width, input_block_height));

				// 画像を直列に変換
				{
					float *fptr = input_blob->mutable_cpu_data() + (input_block_plane_size * n);
					const float *uptr = (const float *)someimg.data;

					const auto Line = someimg.step1();

					if (someimg.channels() == 1)
					{
						if (input_block_width == Line)
							memcpy(fptr, uptr, input_block_width * input_block_height * sizeof(float));
						else
						{
							for (int i = 0; i < input_block_height; i++)
								memcpy(fptr + i * input_block_width, uptr + i * Line, input_block_width * sizeof(float));
						}
					}
					else
					{
						const auto LinePixel = someimg.step1() / someimg.channels();
						const auto Channel = someimg.channels();
						const auto Width = someimg.size().width;
						const auto Height = someimg.size().height;

						for (int i = 0; i < Height; i++)
						{
							for (int j = 0; j < Width; j++)
							{
								for (int ch = 0; ch < Channel; ch++)
								{
									const size_t IndexSrc = i * someimg.step1() + j * Channel + ch;
									const size_t IndexDst = (ch * Height + i) * Width + j;
									fptr[IndexDst] = uptr[IndexSrc];
								}
							}
						}
					}
				}
			}

			assert(input_blob->count() == input_block_plane_size * processNum);

			// 計算
			auto out = mNet->Forward();

			auto b = out[0];

			assert(b->count() == output_block_plane_size * processNum);

			const float *ptr = nullptr;

			if (caffe::Caffe::mode() == caffe::Caffe::CPU)
				ptr = b->cpu_data();
			else
				ptr = b->gpu_data();

			caffe::caffe_copy(output_block_plane_size * processNum, ptr, outputBlockBuf);

			for (int n = 0; n < processNum; n++)
			{
				const int wn = (num + n) % WidthNum;
				const int hn = (num + n) / WidthNum;

				const int w = wn * output_crop_block_width;
				const int h = hn * output_crop_block_height;

				const float *fptr = outputBlockBuf + (output_block_plane_size * n);

				const auto Line = outim.step1();

				// 結果を出力画像にコピー
				if (outim.channels() == 1)
				{
					for (int i = 0; i < output_crop_block_height; i++)
						memcpy(imptr + (h + i) * Line + w, fptr + (i + output_crop_h) * output_block_width + output_crop_w, output_crop_block_width * sizeof(float));
				}
				else
				{
					const auto LinePixel = Line / outim.channels();
					const auto Channel = outim.channels();

					for (int i = 0; i < output_crop_block_height; i++)
					{
						for (int j = 0; j < output_crop_block_width; j++)
						{
							for (int ch = 0; ch < Channel; ch++)
							{
								const size_t IndexSrc = (ch * output_block_height + i + output_crop_h) * output_block_width + j + output_crop_w;
								const size_t IndexDst = ((h + i) * LinePixel + (w + j)) * Channel + ch;

								imptr[IndexDst] = fptr[IndexSrc];
							}
						}
					}
				}

				//{
				//	cv::Mat testim(output_block_size, output_block_size, CV_32FC1);
				//	float *p = (float *)testim.data;
				//	for (int i = 0; i < output_block_size; i++)
				//	{
				//		for (int j = 0; j < output_block_size; j++)
				//		{
				//			p[testim.step1() * i + j] = fptr[i * output_block_size + j];
				//		}
				//	}

				//	const int cv_depth = DepthBitToCVDepth(8);
				//	const double max_val = GetValumeMaxFromCVDepth(cv_depth);
				//	const double eps = GetEPS(cv_depth);

				//	cv::Mat write_iamge;
				//	testim.convertTo(write_iamge, cv_depth, max_val, eps);

				//	cv::imwrite("ti.png", write_iamge);
				//	testim.release();
				//}
			}
		}
	}
	catch (...)
	{
		return Waifu2x::eWaifu2xError_FailedProcessCaffe;
	}

	// 値を0～1にクリッピング
	cv::threshold(outim, outim, 1.0, 1.0, cv::THRESH_TRUNC);
	cv::threshold(outim, outim, 0.0, 0.0, cv::THRESH_TOZERO);

	outMat = outim;

	return Waifu2x::eWaifu2xError_OK;
}

std::string cNet::GetModelName(const boost::filesystem::path &info_path)
{
	Waifu2x::eWaifu2xError ret;

	stInfo info;
	ret = GetInfo(info_path, info);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return std::string();

	return info.name;
}
