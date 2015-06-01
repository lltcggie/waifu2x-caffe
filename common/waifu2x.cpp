#include "waifu2x.h"
#include <caffe/caffe.hpp>
#include <cudnn.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <tclap/CmdLine.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>

#if defined(WIN32) || defined(WIN64)
#include <Windows.h>
#endif

#ifdef _MSC_VER
#ifdef _DEBUG
#pragma comment(lib, "libcaffed.lib")
#pragma comment(lib, "libprotobufd.lib")
#else
#pragma comment(lib, "libcaffe.lib")
#pragma comment(lib, "libprotobuf.lib")
#endif
#pragma comment(lib, "libprotoc.lib")
#endif

const auto block_size = 128;
const auto offset = 0;
const auto layer_num = 7;
const auto output_size = block_size - offset * 2;

const int ConvertMode = CV_RGB2YUV;
const int ConvertInverseMode = CV_YUV2RGB;

std::once_flag waifu2x_once_flag;
std::once_flag waifu2x_cudnn_once_flag;


// cuDNNが使えるかチェック。現状Windowsのみ
bool can_use_cuDNN()
{
	static bool cuDNNFlag = false;
	std::call_once(waifu2x_cudnn_once_flag, [&]()
	{
#if defined(WIN32) || defined(WIN64)
		HMODULE hModule = LoadLibrary(TEXT("cudnn64_65.dll"));
		if (hModule != NULL)
		{
			typedef cudnnStatus_t(*cudnnCreateType)(cudnnHandle_t *);
			typedef cudnnStatus_t(*cudnnDestroyType)(cudnnHandle_t);

			cudnnCreateType cudnnCreateFunc = (cudnnCreateType)GetProcAddress(hModule, "cudnnCreate");
			cudnnDestroyType cudnnDestroyFunc = (cudnnDestroyType)GetProcAddress(hModule, "cudnnDestroy");
			if (cudnnCreateFunc != nullptr && cudnnDestroyFunc != nullptr)
			{
				cudnnHandle_t h;
				if (cudnnCreateFunc(&h) == CUDNN_STATUS_SUCCESS)
				{
					if (cudnnDestroyFunc(h) == CUDNN_STATUS_SUCCESS)
						cuDNNFlag = true;
				}
			}

			FreeLibrary(hModule);
		}
#endif
	});

	return cuDNNFlag;
}

// 画像を読み込んで値を0.0f～1.0fの範囲に変換
eWaifu2xError LoadImage(cv::Mat &float_image, const std::string &input_file)
{
	cv::Mat original_image = cv::imread(input_file, cv::IMREAD_UNCHANGED);
	if (original_image.empty())
		return eWaifu2xError_FailedOpenInputFile;

	cv::Mat convert;
	original_image.convertTo(convert, CV_32F, 1.0 / 255.0);
	original_image.release();

	if (convert.channels() == 1)
		cv::cvtColor(convert, convert, cv::COLOR_GRAY2BGR);
	else if (convert.channels() == 4)
	{
		// アルファチャンネル付きだったら背景を1(白)として画像合成する

		std::vector<cv::Mat> planes;
		cv::split(convert, planes);

		cv::Mat w2 = planes[3];
		cv::Mat w1 = 1.0 - planes[3];

		planes[0] = planes[0].mul(w2) + w1;
		planes[1] = planes[1].mul(w2) + w1;
		planes[2] = planes[2].mul(w2) + w1;

		cv::merge(planes, convert);
	}

	float_image = convert;

	return eWaifu2xError_OK;
}

// 画像から輝度の画像を取り出す
eWaifu2xError CreateBrightnessImage(const cv::Mat &float_image, cv::Mat &im)
{
	cv::Mat converted_color;
	cv::cvtColor(float_image, converted_color, ConvertMode);

	std::vector<cv::Mat> planes;
	cv::split(converted_color, planes);

	im = planes[0];
	planes.clear();

	return eWaifu2xError_OK;
}

// 入力画像の(Photoshopでいう)キャンバスサイズをoutput_sizeの倍数に変更
// 画像は左上配置、余白はcv::BORDER_REPLICATEで埋める
eWaifu2xError PaddingImage(const cv::Mat &input, cv::Mat &output)
{
	const auto h_blocks = (int)floor(input.size().width / output_size) + (input.size().width % output_size == 0 ? 0 : 1);
	const auto w_blocks = (int)floor(input.size().height / output_size) + (input.size().height % output_size == 0 ? 0 : 1);
	const auto height = offset + h_blocks * output_size + offset;
	const auto width = offset + w_blocks * output_size + offset;
	const auto pad_h1 = offset;
	const auto pad_w1 = offset;
	const auto pad_h2 = (height - offset) - input.size().width;
	const auto pad_w2 = (width - offset) - input.size().height;

	cv::copyMakeBorder(input, output, pad_w1, pad_w2, pad_h1, pad_h2, cv::BORDER_REPLICATE);

	return eWaifu2xError_OK;
}

// 画像をcv::INTER_NEARESTで二倍に拡大して、PaddingImage()でパディングする
eWaifu2xError Zoom2xAndPaddingImage(const cv::Mat &input, cv::Mat &output, cv::Size_<int> &zoom_size)
{
	zoom_size = input.size();
	zoom_size.width *= 2;
	zoom_size.height *= 2;

	cv::Mat zoom_image;
	cv::resize(input, zoom_image, zoom_size, 0.0, 0.0, cv::INTER_NEAREST);

	return PaddingImage(zoom_image, output);
}

// 入力画像をzoom_sizeの大きさにcv::INTER_CUBICで拡大し、色情報のみを残す
eWaifu2xError CreateZoomColorImage(const cv::Mat &float_image, const cv::Size_<int> &zoom_size, std::vector<cv::Mat> &cubic_planes)
{
	cv::Mat zoom_cubic_image;
	cv::resize(float_image, zoom_cubic_image, zoom_size, 0.0, 0.0, cv::INTER_CUBIC);

	cv::Mat converted_cubic_image;
	cv::cvtColor(zoom_cubic_image, converted_cubic_image, ConvertMode);
	zoom_cubic_image.release();

	cv::split(converted_cubic_image, cubic_planes);
	converted_cubic_image.release();

	// このY成分は使わないので解放
	cubic_planes[0].release();

	return eWaifu2xError_OK;
}

// 学習したパラメータをファイルから読み込む
eWaifu2xError LoadParameter(boost::shared_ptr<caffe::Net<float>> net, const std::string &param_path)
{
	rapidjson::Document d;
	std::vector<char> jsonBuf;

	try
	{
		FILE *fp = fopen(param_path.c_str(), "rb");
		if (fp == nullptr)
			return eWaifu2xError_FailedOpenModelFile;

		fseek(fp, 0, SEEK_END);
		const auto size = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		jsonBuf.resize(size + 1);
		fread(jsonBuf.data(), 1, size, fp);

		fclose(fp);

		jsonBuf[jsonBuf.size() - 1] = '\0';

		d.Parse(jsonBuf.data());
	}
	catch (...)
	{
		return eWaifu2xError_FailedParseModelFile;
	}

	std::vector<boost::shared_ptr<caffe::Layer<float>>> list;
	auto &v = net->layers();
	for (auto &l : v)
	{
		auto lk = l->type();
		auto &bv = l->blobs();
		if (bv.size() > 0)
			list.push_back(l);
	}

	try
	{
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
				return eWaifu2xError_FailedConstructModel;

			if (!(b1->count() == bias.Size()))
				return eWaifu2xError_FailedConstructModel;

			size_t weightCount = 0;
			std::vector<float> weightList;
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

			std::vector<float> biasList;
			for (auto it2 = bias.Begin(); it2 != bias.End(); ++it2)
				biasList.push_back((float)it2->GetDouble());

			caffe::caffe_copy(b1->count(), biasList.data(), b1Ptr);

			count++;
		}
	}
	catch (...)
	{
		return eWaifu2xError_FailedConstructModel;
	}

	return eWaifu2xError_OK;
}

// モデルファイルからネットワークを構築
// processでcudnnが指定されなかった場合はcuDNNが呼び出されないように変更する
eWaifu2xError ConstractNet(boost::shared_ptr<caffe::Net<float>> &net, const std::string &model_path, const std::string &process)
{
	caffe::NetParameter param;
	if (!caffe::ReadProtoFromTextFile(model_path, &param))
		return eWaifu2xError_FailedOpenModelFile;

	param.mutable_state()->set_phase(caffe::TEST);

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

	net = boost::shared_ptr<caffe::Net<float>>(new caffe::Net<float>(param));

	return eWaifu2xError_OK;
}

// ネットワークを使って画像を再構築する
eWaifu2xError ReconstructImage(boost::shared_ptr<caffe::Net<float>> net, cv::Mat &im, const waifu2xProgressFunc func)
{
	const auto Height = im.size().height;
	const auto Width = im.size().width;
	const auto Line = im.step1();

	assert(im.channels() == 1);

	float *imptr = (float *)im.data;

	try
	{
		const auto input_layer =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
			net->layer_by_name("image_input_layer"));
		assert(input_layer);

		const auto conv7_layer =
			boost::dynamic_pointer_cast<caffe::ConvolutionLayer<float>>(
			net->layer_by_name("conv7_layer"));
		assert(conv7_layer);

		// ネットワークに入力する画像のサイズ(出力画像の幅はlayer_num * 2だけ小さくなる)
		const int block_width = block_size + layer_num * 2;

		std::vector<float> block(block_width * block_width, 0.0f);
		std::vector<float> dummy_data(block.size(), 0.0f);

		// 画像は(消費メモリの都合上)output_size*output_sizeに分けて再構築する
		for (int h = 0; h < Height; h += output_size)
		{
			for (int w = 0; w < Width; w += output_size)
			{
				if (w + block_size <= Width && h + block_size <= Height)
				{
					{
						cv::Mat someimg = im(cv::Rect(w, h, block_size, block_size));
						cv::Mat someborderimg;
						// 画像を中央にパディング。余白はcv::BORDER_REPLICATEで埋める
						cv::copyMakeBorder(someimg, someborderimg, layer_num, layer_num, layer_num, layer_num, cv::BORDER_REPLICATE);
						someimg.release();

						// 画像を直列に変換
						{
							float *fptr = block.data();
							const float *uptr = (const float *)someborderimg.data;

							const auto Line = someborderimg.step1();

							for (int i = 0; i < block_width; i++)
								memcpy(fptr + i * block_width, uptr + i * Line, block_width * sizeof(float));
						}
					}

					// ネットワークに画像を入力
					input_layer->Reset(block.data(), dummy_data.data(), block.size());

					// 計算
					auto out = net->ForwardPrefilled(nullptr);

					auto b = out[0];

					assert(b->count() == block_size * block_size);

					const float *ptr = nullptr;

					if (caffe::Caffe::mode() == caffe::Caffe::CPU)
						ptr = b->cpu_data();
					else
						ptr = b->gpu_data();

					// 結果を入力画像にコピー(後に処理する部分とここで上書きする部分は被らないから、入力画像を上書きしても大丈夫)
					for (int i = 0; i < block_size; i++)
						caffe::caffe_copy(block_size, ptr + i * block_size, imptr + (h + i) * Line + w);
				}
			}
		}
	}
	catch (...)
	{
		return eWaifu2xError_FailedProcessCaffe;
	}

	return eWaifu2xError_OK;
}

eWaifu2xError waifu2x(int argc, char** argv, const std::vector<InputOutputPathPair> &file_paths,
	const std::string &mode, const int noise_level, const double scale_ratio, const std::string &model_dir, const std::string &process,
	std::vector<PathAndErrorPair> &errors, const waifu2xCancelFunc cancel_func, const waifu2xProgressFunc progress_func, const waifu2xTimeFunc time_func)
{
	if (scale_ratio <= 0.0)
		return eWaifu2xError_InvalidParameter;

	const auto StartTime = std::chrono::system_clock::now();

	eWaifu2xError ret;

	std::call_once(waifu2x_once_flag, [argc, argv]()
	{
		assert(argc >= 1);

		int tmpargc = 1;
		char* tmpargvv[] = { argv[0] };
		char** tmpargv = tmpargvv;
		// glog等の初期化
		caffe::GlobalInit(&tmpargc, &tmpargv);
	});

	const auto cuDNNCheckStartTime = std::chrono::system_clock::now();

	std::string process_fix(process);
	if (process_fix == "gpu")
	{
		// cuDNNが使えそうならcuDNNを使う
		if (can_use_cuDNN())
			process_fix = "cudnn";
	}

	const auto cuDNNCheckEndTime = std::chrono::system_clock::now();

	boost::filesystem::path mode_dir_path(model_dir);
	if (!mode_dir_path.is_absolute()) // model_dirが相対パスなら絶対パスに直す
	{
		// まずはカレントディレクトリ下にあるか探す
		mode_dir_path = boost::filesystem::absolute(model_dir);
		if (!boost::filesystem::exists(mode_dir_path) && argc >= 1) // 無かったらargv[0]から実行ファイルのあるフォルダを推定し、そのフォルダ下にあるか探す
		{
			boost::filesystem::path a0(argv[0]);
			if (a0.is_absolute())
				mode_dir_path = a0.branch_path() / model_dir;
		}
	}

	if (!boost::filesystem::exists(mode_dir_path))
		return eWaifu2xError_FailedOpenModelFile;

	if (process_fix == "cpu")
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
	else
		caffe::Caffe::set_mode(caffe::Caffe::GPU);

	boost::shared_ptr<caffe::Net<float>> net_noise;
	boost::shared_ptr<caffe::Net<float>> net_scale;

	if (mode == "noise" || mode == "noise_scale" || mode == "auto_scale")
	{
		const std::string model_path = (mode_dir_path / "srcnn.prototxt").string();
		const std::string param_path = (mode_dir_path / ("noise" + std::to_string(noise_level) + "_model.json")).string();

		ret = ConstractNet(net_noise, model_path, process_fix);
		if (ret != eWaifu2xError_OK)
			return ret;

		ret = LoadParameter(net_noise, param_path);
		if (ret != eWaifu2xError_OK)
			return ret;
	}

	if (mode == "scale" || mode == "noise_scale" || mode == "auto_scale")
	{
		const std::string model_path = (mode_dir_path / "srcnn.prototxt").string();
		const std::string param_path = (mode_dir_path / "scale2.0x_model.json").string();

		ret = ConstractNet(net_scale, model_path, process_fix);
		if (ret != eWaifu2xError_OK)
			return ret;

		ret = LoadParameter(net_scale, param_path);
		if (ret != eWaifu2xError_OK)
			return ret;
	}

	const auto InitEndTime = std::chrono::system_clock::now();

	int fileCount = 0;
	for (const auto &p : file_paths)
	{
		if (progress_func)
			progress_func(file_paths.size(), fileCount);

		if (cancel_func && cancel_func())
			return eWaifu2xError_Cancel;

		const auto &input_file = p.first;
		const auto &output_file = p.second;

		cv::Mat float_image;
		ret = LoadImage(float_image, input_file);
		if (ret != eWaifu2xError_OK)
		{
			errors.emplace_back(p, ret);
			continue;
		}

		cv::Mat im;
		CreateBrightnessImage(float_image, im);

		cv::Size_<int> image_size = im.size();

		const boost::filesystem::path ip(input_file);
		const boost::filesystem::path ipext(ip.extension());

		const bool isJpeg = boost::iequals(ipext.string(), ".jpg") || boost::iequals(ipext.string(), ".jpeg");

		const bool isReconstructNoise = mode == "noise" || mode == "noise_scale" || (mode == "auto_scale" && isJpeg);
		const bool isReconstructScale = mode == "scale" || mode == "noise_scale";

		if (isReconstructNoise)
		{
			PaddingImage(im, im);

			ret = ReconstructImage(net_noise, im, progress_func);
			if (ret != eWaifu2xError_OK)
			{
				errors.emplace_back(p, ret);
				continue;
			}

			// パディングを取り払う
			im = im(cv::Rect(offset, offset, image_size.width, image_size.height));
		}

		if (cancel_func && cancel_func())
			return eWaifu2xError_Cancel;

		const int scale2 = ceil(log2(scale_ratio));
		const double shrinkRatio = scale_ratio / std::pow(2.0, (double)scale2);

		if (isReconstructScale)
		{
			bool isError = false;
			for (int i = 0; i < scale2; i++)
			{
				Zoom2xAndPaddingImage(im, im, image_size);

				ret = ReconstructImage(net_scale, im, progress_func);
				if (ret != eWaifu2xError_OK)
				{
					errors.emplace_back(p, ret);
					isError = true;
					break;
				}

				// パディングを取り払う
				im = im(cv::Rect(offset, offset, image_size.width, image_size.height));
			}

			if (isError)
				continue;
		}

		if (cancel_func && cancel_func())
			return eWaifu2xError_Cancel;

		// 再構築した輝度画像とCreateZoomColorImage()で作成した色情報をマージして通常の画像に変換し、書き込む

		std::vector<cv::Mat> color_planes;
		CreateZoomColorImage(float_image, image_size, color_planes);

		cv::Mat alpha;
		if (float_image.channels() == 4)
		{
			std::vector<cv::Mat> planes;
			cv::split(float_image, planes);
			alpha = planes[3];

			cv::resize(alpha, alpha, image_size, 0.0, 0.0, cv::INTER_CUBIC);
		}

		float_image.release();

		color_planes[0] = im;
		im.release();

		cv::Mat converted_image;
		cv::merge(color_planes, converted_image);
		color_planes.clear();

		cv::Mat process_image;
		cv::cvtColor(converted_image, process_image, ConvertInverseMode);
		converted_image.release();

		// アルファチャンネルがあったら、アルファを付加してカラーからアルファの影響を抜く
		if (!alpha.empty())
		{
			std::vector<cv::Mat> planes;
			cv::split(process_image, planes);
			process_image.release();

			planes.push_back(alpha);

			cv::Mat w2 = planes[3];

			planes[0] = (planes[0] - 1.0).mul(1.0 / w2) + 1.0;
			planes[1] = (planes[1] - 1.0).mul(1.0 / w2) + 1.0;
			planes[2] = (planes[2] - 1.0).mul(1.0 / w2) + 1.0;

			cv::merge(planes, process_image);
		}

		const cv::Size_<int> ns(image_size.width * shrinkRatio, image_size.height * shrinkRatio);
		if (image_size.width != ns.width || image_size.height != ns.height)
			cv::resize(process_image, process_image, ns, 0.0, 0.0, cv::INTER_LINEAR);

		cv::Mat write_iamge;
		process_image.convertTo(write_iamge, CV_8U, 255.0);
		process_image.release();

		if (!cv::imwrite(output_file, write_iamge))
		{
			errors.emplace_back(p, eWaifu2xError_FailedOpenOutputFile);
			continue;
		}

		write_iamge.release();

		fileCount++;
	}

	if (progress_func)
		progress_func(file_paths.size(), fileCount);

	const auto ProcessEndTime = std::chrono::system_clock::now();

	const auto cuDNNCheckTime = (cuDNNCheckEndTime - cuDNNCheckStartTime);
	const auto InitTime = (InitEndTime - StartTime) - cuDNNCheckTime;
	const auto ProcessTime = (ProcessEndTime - InitEndTime);
	if (time_func)
		time_func(std::chrono::duration_cast<std::chrono::milliseconds>(InitTime).count()
		, std::chrono::duration_cast<std::chrono::milliseconds>(cuDNNCheckTime).count()
		, std::chrono::duration_cast<std::chrono::milliseconds>(ProcessTime).count(), process_fix);

	return eWaifu2xError_OK;
}
