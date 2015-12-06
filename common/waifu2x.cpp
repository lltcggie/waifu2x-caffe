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
#include <cuda_runtime.h>

#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#ifdef _MSC_VER
#include <io.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#if defined(WIN32) || defined(WIN64)
#include <Windows.h>
#endif

#ifdef _MSC_VER
#ifdef _DEBUG
#pragma comment(lib, "caffe-d.lib")
#pragma comment(lib, "proto-d.lib")
#pragma comment(lib, "libboost_system-vc120-mt-gd-1_59.lib")
#pragma comment(lib, "libboost_thread-vc120-mt-gd-1_59.lib")
#pragma comment(lib, "libboost_filesystem-vc120-mt-gd-1_59.lib")
#pragma comment(lib, "glogd.lib")
#pragma comment(lib, "gflagsd.lib")
#pragma comment(lib, "libprotobufd.lib")
#pragma comment(lib, "libhdf5_hl_D.lib")
#pragma comment(lib, "libhdf5_D.lib")
#pragma comment(lib, "zlibstaticd.lib")
#pragma comment(lib, "libopenblas.dll.a")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

#pragma comment(lib, "opencv_calib3d249d.lib")
#pragma comment(lib, "opencv_contrib249d.lib")
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
#pragma comment(lib, "libboost_iostreams-vc120-mt-gd-1_59.lib")
#else
#pragma comment(lib, "caffe.lib")
#pragma comment(lib, "proto.lib")
#pragma comment(lib, "libboost_system-vc120-mt-1_59.lib")
#pragma comment(lib, "libboost_thread-vc120-mt-1_59.lib")
#pragma comment(lib, "libboost_filesystem-vc120-mt-1_59.lib")
#pragma comment(lib, "glog.lib")
#pragma comment(lib, "gflags.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libhdf5_hl.lib")
#pragma comment(lib, "libhdf5.lib")
#pragma comment(lib, "zlibstatic.lib")
#pragma comment(lib, "libopenblas.dll.a")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

#pragma comment(lib, "opencv_calib3d249.lib")
#pragma comment(lib, "opencv_contrib249.lib")
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "libboost_iostreams-vc120-mt-1_59.lib")
#endif
#endif

// 入力画像のオフセット
const int offset = 0;
// srcnn.prototxtで定義されたレイヤーの数
const int layer_num = 7;

const int ConvertMode = CV_RGB2YUV;
const int ConvertInverseMode = CV_YUV2RGB;

// 最低限必要なCUDAドライバーのバージョン
const int MinCudaDriverVersion = 6050;

// floatな画像をuint8_tな画像に変換する際の四捨五入に使う値
// https://github.com/nagadomi/waifu2x/commit/797b45ae23665a1c5e3c481c018e48e6f0d0e383
const double clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5);

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

static std::once_flag waifu2x_once_flag;
static std::once_flag waifu2x_cudnn_once_flag;
static std::once_flag waifu2x_cuda_once_flag;

#ifndef CUDA_CHECK_WAIFU2X
#define CUDA_CHECK_WAIFU2X(condition) \
 do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) throw error; \
 } while (0)
#endif

#define CUDA_HOST_SAFE_FREE(ptr) \
	do { \
		if (ptr) { \
			cudaFreeHost(ptr); \
			ptr = nullptr; \
		} \
	} while (0)

#define SAFE_DELETE_WAIFU2X(ptr) \
	do { \
		if (ptr) { \
			delete [] ptr; \
			ptr = nullptr; \
		} \
	} while (0)

namespace
{
	class IgnoreErrorCV
	{
	private:
		static int handleError(int status, const char* func_name,
			const char* err_msg, const char* file_name,
			int line, void* userdata)
		{
			return 0;
		}

	public:
		IgnoreErrorCV()
		{
			cv::redirectError(handleError);
		}
	};

	IgnoreErrorCV g_IgnoreErrorCV;
}

template<typename BufType>
static bool writeFile(boost::iostreams::stream<boost::iostreams::file_descriptor> &os, const std::vector<BufType> &buf)
{
	if (!os)
		return false;

	const auto WriteSize = sizeof(BufType) * buf.size();
	os.write((const char *)buf.data(), WriteSize);
	if (os.fail())
		return false;

	return true;
}

template<typename BufType>
static bool writeFile(const boost::filesystem::path &path, std::vector<BufType> &buf)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor> os(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

	return writeFile(os, buf);
}

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
	boost::iostreams::stream<boost::iostreams::file_descriptor_source> is(path, std::ios_base::in | std::ios_base::binary);

	return readFile(is, buf);
}

static Waifu2x::eWaifu2xError readProtoText(const boost::filesystem::path &path, ::google::protobuf::Message* proto)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor_source> is(path, std::ios_base::in);

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

static Waifu2x::eWaifu2xError readProtoBinary(const boost::filesystem::path &path, ::google::protobuf::Message* proto)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor_source> is(path, std::ios_base::in | std::ios_base::binary);

	if (!is)
		return Waifu2x::eWaifu2xError_FailedParseModelFile;

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

static Waifu2x::eWaifu2xError writeProtoBinary(const ::google::protobuf::Message& proto, const boost::filesystem::path &path)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor> os(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

	if (!os)
		return Waifu2x::eWaifu2xError_FailedWriteModelFile;

	if (!proto.SerializePartialToOstream(&os))
		return Waifu2x::eWaifu2xError_FailedWriteModelFile;

	return Waifu2x::eWaifu2xError_OK;
}


Waifu2x::Waifu2x() : is_inited(false), isCuda(false), input_block(nullptr), dummy_data(nullptr), output_block(nullptr)
{
}

Waifu2x::~Waifu2x()
{
	destroy();
}

// cuDNNが使えるかチェック。現状Windowsのみ
Waifu2x::eWaifu2xcuDNNError Waifu2x::can_use_cuDNN()
{
	static eWaifu2xcuDNNError cuDNNFlag = eWaifu2xcuDNNError_NotFind;
	std::call_once(waifu2x_cudnn_once_flag, [&]()
	{
#if defined(WIN32) || defined(WIN64)
		HMODULE hModule = LoadLibrary(TEXT(CUDNN_DLL_NAME));
		if (hModule != NULL)
		{
			typedef cudnnStatus_t(__stdcall * cudnnCreateType)(cudnnHandle_t *);
			typedef cudnnStatus_t(__stdcall * cudnnDestroyType)(cudnnHandle_t);
			typedef uint64_t(__stdcall * cudnnGetVersionType)();

			cudnnCreateType cudnnCreateFunc = (cudnnCreateType)GetProcAddress(hModule, "cudnnCreate");
			cudnnDestroyType cudnnDestroyFunc = (cudnnDestroyType)GetProcAddress(hModule, "cudnnDestroy");
			cudnnGetVersionType cudnnGetVersionFunc = (cudnnGetVersionType)GetProcAddress(hModule, "cudnnGetVersion");
			if (cudnnCreateFunc != nullptr && cudnnDestroyFunc != nullptr && cudnnGetVersionFunc != nullptr)
			{
				if (cudnnGetVersionFunc() >= 2000)
				{
					cudnnHandle_t h;
					if (cudnnCreateFunc(&h) == CUDNN_STATUS_SUCCESS)
					{
						if (cudnnDestroyFunc(h) == CUDNN_STATUS_SUCCESS)
							cuDNNFlag = eWaifu2xcuDNNError_OK;
						else
							cuDNNFlag = eWaifu2xcuDNNError_CannotCreate;
					}
					else
						cuDNNFlag = eWaifu2xcuDNNError_CannotCreate;
				}
				else
					cuDNNFlag = eWaifu2xcuDNNError_OldVersion;
			}
			else
				cuDNNFlag = eWaifu2xcuDNNError_NotFind;

			FreeLibrary(hModule);
		}
#endif
	});

	return cuDNNFlag;
}

// CUDAが使えるかチェック
Waifu2x::eWaifu2xCudaError Waifu2x::can_use_CUDA()
{
	static eWaifu2xCudaError CudaFlag = eWaifu2xCudaError_NotFind;
	std::call_once(waifu2x_cuda_once_flag, [&]()
	{
		int driverVersion = 0;
		if (cudaDriverGetVersion(&driverVersion) == cudaSuccess)
		{
			if (driverVersion > 0)
			{
				int runtimeVersion;
				if (cudaRuntimeGetVersion(&runtimeVersion) == cudaSuccess)
				{
					if (runtimeVersion >= MinCudaDriverVersion && driverVersion >= runtimeVersion)
						CudaFlag = eWaifu2xCudaError_OK;
					else
						CudaFlag = eWaifu2xCudaError_OldVersion;
				}
				else
					CudaFlag = eWaifu2xCudaError_NotFind;
			}
			else
				CudaFlag = eWaifu2xCudaError_NotFind;
		}
		else
			CudaFlag = eWaifu2xCudaError_NotFind;
	});

	return CudaFlag;
}

void Waifu2x::init_liblary()
{
}

void Waifu2x::quit_liblary()
{
}

cv::Mat Waifu2x::LoadMat(const boost::filesystem::path &path)
{
	cv::Mat mat;
	LoadMat(mat, path);

	return mat;
}

Waifu2x::eWaifu2xError Waifu2x::AlphaMakeBorder(std::vector<cv::Mat> &planes, const cv::Mat &alpha, const int offset)
{
	// このカーネルと画像の畳込みを行うと、(x, y)を中心とした3×3領域の合計値が求まる
	const static cv::Mat sum2d_kernel = (cv::Mat_<double>(3, 3) <<
		1., 1., 1.,
		1., 1., 1.,
		1., 1., 1.);

	cv::Mat mask;
	cv::threshold(alpha, mask, 0.0, 1.0, cv::THRESH_BINARY); // アルファチャンネルを二値化してマスクとして扱う

	cv::Mat mask_nega;
	cv::threshold(mask, mask_nega, 0.0, 1.0, cv::THRESH_BINARY_INV); // 反転したマスク（値が1の箇所は完全透明でない有効な画素となる）

	for (auto &p : planes) // 完全に透明なピクセルにあるゴミを取る
	{
		p = p.mul(mask);
	}

	for (int i = 0; i < offset; i++)
	{
		cv::Mat mask_weight;
		cv::filter2D(mask, mask_weight, -1, sum2d_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT); // マスクの3×3領域の合計値を求める

		cv::Mat mask_nega_u8;
		mask_nega.convertTo(mask_nega_u8, CV_8U, 255.0, clip_eps8); // mask_negaのCV_U8版（OpenCVのAPI上必要になる）

		for (auto &p : planes) // 1チャンネルずつ処理
		{
			// チャンネルの3×3領域内の有効画素の平均値を求める
			cv::Mat border;
			cv::filter2D(p, border, -1, sum2d_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
			border /= mask_weight;

			// チャンネルの有効な画素の部分に、計算した平均値をコピー
			border.copyTo(p, mask_nega_u8);
		}

		// マスクを1回膨張させたものを新しいマスクとする(マスクの3×3領域の合計値を求めたものの非0領域は、マスクを1回膨張させたものの領域に等しい)
		cv::threshold(mask_weight, mask, 0.0, 1.0, cv::THRESH_BINARY);
		// 新しいマスクの反転したマスクを計算
		cv::threshold(mask, mask_nega, 0.0, 1.0, cv::THRESH_BINARY_INV);
	}

	// 画素を0から1にクリッピング
	for (auto &p : planes)
	{
		cv::threshold(p, p, 1.0, 1.0, cv::THRESH_TRUNC);
		cv::threshold(p, p, 0.0, 0.0, cv::THRESH_TOZERO);
	}

	return eWaifu2xError_OK;
}

// 画像を読み込んで値を0.0f～1.0fの範囲に変換
Waifu2x::eWaifu2xError Waifu2x::LoadMat(cv::Mat &float_image, const boost::filesystem::path &input_file)
{
	cv::Mat original_image;

	{
		std::vector<char> img_data;
		if (!readFile(input_file, img_data))
			return Waifu2x::eWaifu2xError_FailedOpenInputFile;

		cv::Mat im(img_data.size(), 1, CV_8U, img_data.data());
		original_image = cv::imdecode(im, cv::IMREAD_UNCHANGED);

		if (original_image.empty())
		{
			const eWaifu2xError ret = LoadMatBySTBI(original_image, img_data);
			if (ret != eWaifu2xError_OK)
				return ret;
		}
	}

	cv::Mat convert;
	original_image.convertTo(convert, CV_32F, 1.0 / 255.0);
	original_image.release();

	if (convert.channels() == 1)
		cv::cvtColor(convert, convert, cv::COLOR_GRAY2BGR);
	else if (convert.channels() == 4)
	{
		// アルファチャンネル付きだったら透明なピクセルのと不透明なピクセルの境界部分の色を広げる

		std::vector<cv::Mat> planes;
		cv::split(convert, planes);

		cv::Mat alpha = planes[3];
		planes.resize(3);
		AlphaMakeBorder(planes, alpha, layer_num);

		planes.push_back(alpha);
		cv::merge(planes, convert);
	}

	float_image = convert;

	return eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::LoadMatBySTBI(cv::Mat &float_image, const std::vector<char> &img_data)
{
	int x, y, comp;
	stbi_uc *data = stbi_load_from_memory((const stbi_uc *)img_data.data(), img_data.size(), &x, &y, &comp, 4);
	if (!data)
		return eWaifu2xError_FailedOpenInputFile;

	int type = 0;
	switch (comp)
	{
	case 1:
	case 3:
	case 4:
		type = CV_MAKETYPE(CV_8U, comp);
		break;

	default:
		return eWaifu2xError_FailedOpenInputFile;
	}

	float_image = cv::Mat(cv::Size(x, y), type);

	const auto LinePixel = float_image.step1() / float_image.channels();
	const auto Channel = float_image.channels();
	const auto Width = float_image.size().width;
	const auto Height = float_image.size().height;

	assert(x == Width);
	assert(y == Height);
	assert(Channel == comp);

	auto ptr = float_image.data;
	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			for (int ch = 0; ch < Channel; ch++)
				ptr[(i * LinePixel + j) * comp + ch] = data[(i * x + j) * comp + ch];
		}
	}

	stbi_image_free(data);

	if (comp >= 3)
	{
		// RGBだからBGRに変換
		for (int i = 0; i < y; i++)
		{
			for (int j = 0; j < x; j++)
				std::swap(ptr[(i * LinePixel + j) * comp + 0], ptr[(i * LinePixel + j) * comp + 2]);
		}
	}

	return eWaifu2xError_OK;
}

// 画像から輝度の画像を取り出す
Waifu2x::eWaifu2xError Waifu2x::CreateBrightnessImage(const cv::Mat &float_image, cv::Mat &im)
{
	if (float_image.channels() > 1)
	{
		cv::Mat converted_color;
		cv::cvtColor(float_image, converted_color, ConvertMode);

		std::vector<cv::Mat> planes;
		cv::split(converted_color, planes);

		im = planes[0];
		planes.clear();
	}
	else
		im = float_image;

	return eWaifu2xError_OK;
}

// 入力画像の(Photoshopでいう)キャンバスサイズをoutput_sizeの倍数に変更
// 画像は左上配置、余白はcv::BORDER_REPLICATEで埋める
Waifu2x::eWaifu2xError Waifu2x::PaddingImage(const cv::Mat &input, cv::Mat &output)
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
Waifu2x::eWaifu2xError Waifu2x::Zoom2xAndPaddingImage(const cv::Mat &input, cv::Mat &output, cv::Size_<int> &zoom_size)
{
	zoom_size = input.size();
	zoom_size.width *= 2;
	zoom_size.height *= 2;

	cv::Mat zoom_image;
	cv::resize(input, zoom_image, zoom_size, 0.0, 0.0, cv::INTER_NEAREST);

	return PaddingImage(zoom_image, output);
}

// 入力画像をzoom_sizeの大きさにcv::INTER_CUBICで拡大し、色情報のみを残す
Waifu2x::eWaifu2xError Waifu2x::CreateZoomColorImage(const cv::Mat &float_image, const cv::Size_<int> &zoom_size, std::vector<cv::Mat> &cubic_planes)
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

// モデルファイルからネットワークを構築
// processでcudnnが指定されなかった場合はcuDNNが呼び出されないように変更する
Waifu2x::eWaifu2xError Waifu2x::ConstractNet(boost::shared_ptr<caffe::Net<float>> &net, const boost::filesystem::path &model_path, const boost::filesystem::path &param_path, const std::string &process)
{
	boost::filesystem::path caffemodel_path = param_path;
	caffemodel_path += ".caffemodel";
	boost::filesystem::path modelbin_path = model_path;
	modelbin_path += ".protobin";

	caffe::NetParameter param;
	if (readProtoBinary(modelbin_path, &param) == eWaifu2xError_OK)
	{
		Waifu2x::eWaifu2xError ret;

		ret = SetParameter(param, process);
		if (ret != eWaifu2xError_OK)
			return ret;

		caffe::NetParameter param_caffemodel;
		ret = readProtoBinary(caffemodel_path, &param_caffemodel);
		if (ret != eWaifu2xError_OK)
			return ret;

		if (!caffe::UpgradeNetAsNeeded(caffemodel_path.string(), &param_caffemodel))
			return Waifu2x::eWaifu2xError_FailedParseModelFile;

		net = boost::shared_ptr<caffe::Net<float>>(new caffe::Net<float>(param));
		net->CopyTrainedLayersFrom(param_caffemodel);

		input_plane = param.input_dim(1);
	}
	else
	{
		const auto ret = LoadParameterFromJson(net, model_path, param_path, modelbin_path, caffemodel_path, process);
		if (ret != eWaifu2xError_OK)
			return ret;
	}

	return eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::SetParameter(caffe::NetParameter &param, const std::string &process) const
{
	param.mutable_state()->set_phase(caffe::TEST);

	{
		auto mid = param.mutable_input_dim();

		if (mid->size() != 4)
			return eWaifu2xError_FailedParseModelFile;

		*mid->Mutable(0) = batch_size;
		*mid->Mutable(2) = input_block_size;
		*mid->Mutable(3) = input_block_size;
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

	return eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::LoadParameterFromJson(boost::shared_ptr<caffe::Net<float>> &net, const boost::filesystem::path &model_path, const boost::filesystem::path &param_path
	, const boost::filesystem::path &modelbin_path, const boost::filesystem::path &caffemodel_path, const std::string &process)
{
	Waifu2x::eWaifu2xError ret;

	caffe::NetParameter param;
	ret = readProtoText(model_path, &param);
	if (ret != eWaifu2xError_OK)
		return ret;

	ret = writeProtoBinary(param, modelbin_path);
	if (ret != eWaifu2xError_OK)
		return ret;

	ret = SetParameter(param, process);
	if (ret != eWaifu2xError_OK)
		return ret;

	net = boost::shared_ptr<caffe::Net<float>>(new caffe::Net<float>(param));

	rapidjson::Document d;
	std::vector<char> jsonBuf;

	try
	{
		boost::iostreams::stream<boost::iostreams::file_descriptor_source> is(param_path, std::ios_base::in | std::ios_base::binary);
		if(!is)
			return eWaifu2xError_FailedOpenModelFile;

		const size_t size = is.seekg(0, std::ios::end).tellg();
		is.seekg(0, std::ios::beg);

		jsonBuf.resize(size + 1);
		is.read(jsonBuf.data(), jsonBuf.size());

		jsonBuf[jsonBuf.size() - 1] = '\0';

		d.Parse(jsonBuf.data());
	}
	catch (...)
	{
		return eWaifu2xError_FailedParseModelFile;
	}

	if (d.Size() != 7)
		return eWaifu2xError_FailedParseModelFile;

	int inputPlane = 0;
	int outputPlane = 0;
	try
	{
		inputPlane = d[0]["nInputPlane"].GetInt();
		outputPlane = d[d.Size() - 1]["nOutputPlane"].GetInt();
	}
	catch (...)
	{
		return eWaifu2xError_FailedParseModelFile;
	}

	if (inputPlane == 0 || outputPlane == 0)
		return eWaifu2xError_FailedParseModelFile;

	if (inputPlane != outputPlane)
		return eWaifu2xError_FailedParseModelFile;

	//if (param.layer_size() < 17)
	//	return eWaifu2xError_FailedParseModelFile;

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
				return eWaifu2xError_FailedConstructModel;

			if (!(b1->count() == bias.Size()))
				return eWaifu2xError_FailedConstructModel;

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

		net->ToProto(&param);

		ret = writeProtoBinary(param, caffemodel_path);
		if (ret != eWaifu2xError_OK)
			return ret;
	}
	catch (...)
	{
		return eWaifu2xError_FailedConstructModel;
	}

	input_plane = inputPlane;

	return eWaifu2xError_OK;
}

// ネットワークを使って画像を再構築する
Waifu2x::eWaifu2xError Waifu2x::ReconstructImage(boost::shared_ptr<caffe::Net<float>> net, cv::Mat &im)
{
	const auto Height = im.size().height;
	const auto Width = im.size().width;
	const auto Line = im.step1();

	assert(Width % output_size == 0);
	assert(Height % output_size == 0);

	assert(im.channels() == 1 || im.channels() == 3);

	cv::Mat outim(im.rows, im.cols, im.type());

	// float *imptr = (float *)im.data;
	float *imptr = (float *)outim.data;

	try
	{
		auto input_blobs = net->input_blobs();
		auto input_blob = net->input_blobs()[0];

		input_blob->Reshape(batch_size, input_plane, input_block_size, input_block_size);

		assert(im.channels() == input_plane);
		assert(input_blob->shape(1) == input_plane);

		const int WidthNum = Width / output_size;
		const int HeightNum = Height / output_size;

		const int BlockNum = WidthNum * HeightNum;

		const int input_block_plane_size = input_block_size * input_block_size * input_plane;
		const int output_block_plane_size = output_block_size * output_block_size * input_plane;

		const int output_padding = inner_padding + outer_padding - layer_num;

		// 画像は(消費メモリの都合上)output_size*output_sizeに分けて再構築する
		for (int num = 0; num < BlockNum; num += batch_size)
		{
			const int processNum = (BlockNum - num) >= batch_size ? batch_size : BlockNum - num;

			if (processNum < batch_size)
				input_blob->Reshape(processNum, input_plane, input_block_size, input_block_size);

			for (int n = 0; n < processNum; n++)
			{
				const int wn = (num + n) % WidthNum;
				const int hn = (num + n) / WidthNum;

				const int w = wn * output_size;
				const int h = hn * output_size;

				if (w + crop_size <= Width && h + crop_size <= Height)
				{
					int x, y;
					x = w - inner_padding;
					y = h - inner_padding;

					int width, height;

					width = crop_size + inner_padding * 2;
					height = crop_size + inner_padding * 2;

					int top, bottom, left, right;

					top = outer_padding;
					bottom = outer_padding;
					left = outer_padding;
					right = outer_padding;

					if (x < 0)
					{
						left += -x;
						width -= -x;
						x = 0;
					}

					if (x + width > Width)
					{
						right += (x + width) - Width;
						width = Width - x;
					}

					if (y < 0)
					{
						top += -y;
						height -= -y;
						y = 0;
					}

					if (y + height > Height)
					{
						bottom += (y + height) - Height;
						height = Height - y;
					}

					cv::Mat someimg = im(cv::Rect(x, y, width, height));

					cv::Mat someborderimg;
					// 画像を中央にパディング。余白はcv::BORDER_REPLICATEで埋める
					// 実はimで画素が存在する部分は余白と認識されないが、inner_paddingがlayer_numでouter_paddingが1以上ならそこの部分の画素は結果画像として取り出す部分には影響しない
					cv::copyMakeBorder(someimg, someborderimg, top, bottom, left, right, cv::BORDER_REPLICATE);
					someimg.release();

					// 画像を直列に変換
					{
						float *fptr = input_block + (input_block_plane_size * n);
						const float *uptr = (const float *)someborderimg.data;

						const auto Line = someborderimg.step1();

						if (someborderimg.channels() == 1)
						{
							if (input_block_size == Line)
								memcpy(fptr, uptr, input_block_size * input_block_size * sizeof(float));
							else
							{
								for (int i = 0; i < input_block_size; i++)
									memcpy(fptr + i * input_block_size, uptr + i * Line, input_block_size * sizeof(float));
							}
						}
						else
						{
							const auto LinePixel = someborderimg.step1() / someborderimg.channels();
							const auto Channel = someborderimg.channels();
							const auto Width = someborderimg.size().width;
							const auto Height = someborderimg.size().height;

							for (int i = 0; i < Height; i++)
							{
								for (int j = 0; j < LinePixel; j++)
								{
									for (int ch = 0; ch < Channel; ch++)
										fptr[(ch * Height + i) * Width + j] = uptr[(i * LinePixel + j) * Channel + ch];
								}
							}
						}
					}
				}
			}

			assert(input_blob->count() == input_block_plane_size * processNum);

			// ネットワークに画像を入力
			input_blob->set_cpu_data(input_block);

			// 計算
			auto out = net->ForwardPrefilled(nullptr);

			auto b = out[0];

			assert(b->count() == output_block_plane_size * processNum);

			const float *ptr = nullptr;

			if (caffe::Caffe::mode() == caffe::Caffe::CPU)
				ptr = b->cpu_data();
			else
				ptr = b->gpu_data();

			caffe::caffe_copy(output_block_plane_size * processNum, ptr, output_block);

			for (int n = 0; n < processNum; n++)
			{
				const int wn = (num + n) % WidthNum;
				const int hn = (num + n) / WidthNum;

				const int w = wn * output_size;
				const int h = hn * output_size;

				const float *fptr = output_block + (output_block_plane_size * n);

				// 結果を出力画像にコピー
				if (outim.channels() == 1)
				{
					for (int i = 0; i < crop_size; i++)
						memcpy(imptr + (h + i) * Line + w, fptr + (i + output_padding) * output_block_size + output_padding, crop_size * sizeof(float));
				}
				else
				{
					const auto LinePixel = outim.step1() / outim.channels();
					const auto Channel = outim.channels();

					for (int i = 0; i < crop_size; i++)
					{
						for (int j = 0; j < crop_size; j++)
						{
							for (int ch = 0; ch < Channel; ch++)
								imptr[((h + i) * LinePixel + (w + j)) * Channel + ch] = fptr[(ch * output_block_size + i + output_padding) * output_block_size + j + output_padding];
						}
					}
				}
			}
		}
	}
	catch (...)
	{
		return eWaifu2xError_FailedProcessCaffe;
	}

	im = outim;

	return eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::init(int argc, char** argv, const std::string &Mode, const int NoiseLevel, const double ScaleRatio, const boost::filesystem::path &ModelDir, const std::string &Process,
	const bool UseTTA, const int CropSize, const int BatchSize)
{
	Waifu2x::eWaifu2xError ret;

	if (is_inited)
		return eWaifu2xError_OK;

	if (ScaleRatio <= 0.0)
		return eWaifu2xError_InvalidParameter;

	try
	{
		mode = Mode;
		noise_level = NoiseLevel;
		scale_ratio = ScaleRatio;
		model_dir = ModelDir;
		process = Process;
		use_tta = UseTTA;

		crop_size = CropSize;
		batch_size = BatchSize;

		inner_padding = layer_num;
		outer_padding = 1;

		output_size = crop_size - offset * 2;
		input_block_size = crop_size + (inner_padding + outer_padding) * 2;
		original_width_height = 128 + layer_num * 2;

		output_block_size = crop_size + (inner_padding + outer_padding - layer_num) * 2;

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

		if (process == "gpu")
		{
			if (can_use_CUDA() != eWaifu2xCudaError_OK)
				return eWaifu2xError_FailedCudaCheck;
			// cuDNNが使えそうならcuDNNを使う
			else if (can_use_cuDNN() == eWaifu2xcuDNNError_OK)
				process = "cudnn";
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

		if (process == "cpu")
		{
			caffe::Caffe::set_mode(caffe::Caffe::CPU);
			isCuda = false;
		}
		else
		{
			caffe::Caffe::set_mode(caffe::Caffe::GPU);
			isCuda = true;
		}

		if (mode == "noise" || mode == "noise_scale" || mode == "auto_scale")
		{
			const boost::filesystem::path model_path = (mode_dir_path / "srcnn.prototxt").string();
			const boost::filesystem::path param_path = (mode_dir_path / ("noise" + std::to_string(noise_level) + "_model.json")).string();

			ret = ConstractNet(net_noise, model_path, param_path, process);
			if (ret != eWaifu2xError_OK)
				return ret;
		}

		if (mode == "scale" || mode == "noise_scale" || mode == "auto_scale")
		{
			const boost::filesystem::path model_path = (mode_dir_path / "srcnn.prototxt").string();
			const boost::filesystem::path param_path = (mode_dir_path / "scale2.0x_model.json").string();

			ret = ConstractNet(net_scale, model_path, param_path, process);
			if (ret != eWaifu2xError_OK)
				return ret;
		}

		const int input_block_plane_size = input_block_size * input_block_size * input_plane;
		const int output_block_plane_size = output_block_size * output_block_size * input_plane;

		if (isCuda)
		{
			CUDA_CHECK_WAIFU2X(cudaHostAlloc(&input_block, sizeof(float) * input_block_plane_size * batch_size, cudaHostAllocWriteCombined));
			CUDA_CHECK_WAIFU2X(cudaHostAlloc(&dummy_data, sizeof(float) * input_block_plane_size * batch_size, cudaHostAllocWriteCombined));
			CUDA_CHECK_WAIFU2X(cudaHostAlloc(&output_block, sizeof(float) * output_block_plane_size * batch_size, cudaHostAllocDefault));
		}
		else
		{
			input_block = new float[input_block_plane_size * batch_size];
			dummy_data = new float[input_block_plane_size * batch_size];
			output_block = new float[output_block_plane_size * batch_size];
		}

		for (size_t i = 0; i < input_block_plane_size * batch_size; i++)
			dummy_data[i] = 0.0f;

		is_inited = true;
	}
	catch (...)
	{
		return eWaifu2xError_InvalidParameter;
	}

	return eWaifu2xError_OK;
}

void Waifu2x::destroy()
{
	net_noise.reset();
	net_scale.reset();

	if (isCuda)
	{
		CUDA_HOST_SAFE_FREE(input_block);
		CUDA_HOST_SAFE_FREE(dummy_data);
		CUDA_HOST_SAFE_FREE(output_block);
	}
	else
	{
		SAFE_DELETE_WAIFU2X(input_block);
		SAFE_DELETE_WAIFU2X(dummy_data);
		SAFE_DELETE_WAIFU2X(output_block);
	}

	is_inited = false;
}

Waifu2x::eWaifu2xError Waifu2x::WriteMat(const cv::Mat &im, const boost::filesystem::path &output_file)
{
	const boost::filesystem::path ip(output_file);
	const std::string ext = ip.extension().string();

	const bool isJpeg = boost::iequals(ext, ".jpg") || boost::iequals(ext, ".jpeg");

	if (boost::iequals(ext, ".tga"))
	{
		unsigned char *data = im.data;

		std::vector<unsigned char> rgbimg;
		if (im.channels() >= 3 || im.step1() != im.size().width * im.channels()) // RGB用バッファにコピー(あるいはパディングをとる)
		{
			const auto Line = im.step1();
			const auto Channel = im.channels();
			const auto Width = im.size().width;
			const auto Height = im.size().height;

			rgbimg.resize(Width * Height * Channel);

			const auto Stride = Width * Channel;
			for (int i = 0; i < Height; i++)
				memcpy(rgbimg.data() + Stride * i, im.data + Line * i, Stride);

			data = rgbimg.data();
		}

		if (im.channels() >= 3) // BGRをRGBに並び替え
		{
			const auto Line = im.step1();
			const auto Channel = im.channels();
			const auto Width = im.size().width;
			const auto Height = im.size().height;

			auto ptr = rgbimg.data();
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
					std::swap(ptr[(i * Width + j) * Channel + 0], ptr[(i * Width + j) * Channel + 2]);
			}
		}

		boost::iostreams::stream<boost::iostreams::file_descriptor> os(output_file, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
		if(!os)
			return eWaifu2xError_FailedOpenOutputFile;

		if (!stbi_write_tga(os, im.size().width, im.size().height, im.channels(), data))
			return eWaifu2xError_FailedOpenOutputFile;

		return eWaifu2xError_OK;
	}

	try
	{
		std::vector<uchar> buf;
		cv::imencode(ext, im, buf);

		if (writeFile(output_file, buf))
			return eWaifu2xError_OK;

	}
	catch (...)
	{
	}

	return eWaifu2xError_FailedOpenOutputFile;
}

Waifu2x::eWaifu2xError Waifu2x::BeforeReconstructFloatMatProcess(const cv::Mat &in, cv::Mat &out, bool &convertBGRflag)
{
	Waifu2x::eWaifu2xError ret;

	convertBGRflag = false;

	cv::Mat im;
	if (input_plane == 1)
		CreateBrightnessImage(in, im);
	else
	{
		im = in;
		if (in.channels() == 1)
		{
			cv::cvtColor(in, im, CV_GRAY2BGR);
			convertBGRflag = true;
		}

		std::vector<cv::Mat> planes;
		cv::split(im, planes);

		if (im.channels() == 4)
			planes.resize(3);

		// BGRからRGBにする
		std::swap(planes[0], planes[2]);

		cv::merge(planes, im);
	}

	out = im;

	return eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::ReconstructFloatMat(const bool isReconstructNoise, const bool isReconstructScale, const waifu2xCancelFunc cancel_func, const cv::Mat &in, cv::Mat &out)
{
	Waifu2x::eWaifu2xError ret;

	cv::Mat im(in);
	cv::Size_<int> image_size = im.size();

	if (isReconstructNoise)
	{
		PaddingImage(im, im);

		ret = ReconstructImage(net_noise, im);
		if (ret != eWaifu2xError_OK)
			return ret;

		// パディングを取り払う
		im = im(cv::Rect(offset, offset, image_size.width, image_size.height));
	}

	if (cancel_func && cancel_func())
		return eWaifu2xError_Cancel;

	const int scale2 = ceil(log2(scale_ratio));

	if (isReconstructScale)
	{
		bool isError = false;
		for (int i = 0; i < scale2; i++)
		{
			Zoom2xAndPaddingImage(im, im, image_size);

			ret = ReconstructImage(net_scale, im);
			if (ret != eWaifu2xError_OK)
				return ret;

			// パディングを取り払う
			im = im(cv::Rect(offset, offset, image_size.width, image_size.height));
		}
	}

	if (cancel_func && cancel_func())
		return eWaifu2xError_Cancel;

	out = im;

	return eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::Reconstruct(const bool isReconstructNoise, const bool isReconstructScale, const waifu2xCancelFunc cancel_func, const cv::Mat &in, cv::Mat &out)
{
	Waifu2x::eWaifu2xError ret;

	bool convertBGRflag = false;
	cv::Mat brfm;
	ret = BeforeReconstructFloatMatProcess(in, brfm, convertBGRflag);
	if (ret != eWaifu2xError_OK)
		return ret;

	cv::Mat reconstruct_image;
	if (!use_tta) // 普通に処理
	{
		ret = ReconstructFloatMat(isReconstructNoise, isReconstructScale, cancel_func, brfm, reconstruct_image);
		if (ret != eWaifu2xError_OK)
			return ret;
	}
	else // Test-Time Augmentation Mode
	{
		const auto RotateClockwise90 = [](cv::Mat &mat)
		{
			cv::transpose(mat, mat);
			cv::flip(mat, mat, 1);
		};

		const auto RotateClockwise90N = [RotateClockwise90](cv::Mat &mat, const int rotateNum)
		{
			for (int i = 0; i < rotateNum; i++)
				RotateClockwise90(mat);
		};

		const auto RotateCounterclockwise90 = [](cv::Mat &mat)
		{
			cv::transpose(mat, mat);
			cv::flip(mat, mat, 0);
		};

		const auto RotateCounterclockwise90N = [RotateCounterclockwise90](cv::Mat &mat, const int rotateNum)
		{
			for (int i = 0; i < rotateNum; i++)
				RotateCounterclockwise90(mat);
		};

		cv::Mat ri[8];
		for (int i = 0; i < 8; i++)
		{
			cv::Mat in(brfm.clone());

			const int rotateNum = i % 4;
			RotateClockwise90N(in, rotateNum);

			if (i >= 4)
				cv::flip(in, in, 1); // 垂直軸反転

			ret = ReconstructFloatMat(isReconstructNoise, isReconstructScale, cancel_func, in, in);
			if (ret != eWaifu2xError_OK)
				return ret;

			if (i >= 4)
				cv::flip(in, in, 1); // 垂直軸反転

			RotateCounterclockwise90N(in, rotateNum);

			ri[i] = in;
		}

		reconstruct_image = ri[0];
		for (int i = 1; i < 8; i++)
			reconstruct_image += ri[i];

		reconstruct_image /= 8.0;
	}

	if (convertBGRflag)
	{
		cv::cvtColor(reconstruct_image, reconstruct_image, CV_RGB2GRAY); // この地点ではまだRGBなことに注意
	}

	out = reconstruct_image;

	return eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::AfterReconstructFloatMatProcess(const bool isReconstructScale, const waifu2xCancelFunc cancel_func, const cv::Mat &floatim, const cv::Mat &in, cv::Mat &out)
{
	cv::Size_<int> image_size = in.size();

	cv::Mat process_image;
	if (input_plane == 1)
	{
		// 再構築した輝度画像とCreateZoomColorImage()で作成した色情報をマージして通常の画像に変換し、書き込む

		std::vector<cv::Mat> color_planes;
		CreateZoomColorImage(floatim, image_size, color_planes);

		color_planes[0] = in;

		cv::Mat converted_image;
		cv::merge(color_planes, converted_image);
		color_planes.clear();

		cv::cvtColor(converted_image, process_image, ConvertInverseMode);
		converted_image.release();
	}
	else
	{
		std::vector<cv::Mat> planes;
		cv::split(in, planes);

		// RGBからBGRに直す
		std::swap(planes[0], planes[2]);

		cv::merge(planes, process_image);
	}

	const int scale2 = ceil(log2(scale_ratio));
	const double shrinkRatio = scale_ratio / std::pow(2.0, (double)scale2);

	cv::Mat alpha;
	if (floatim.channels() == 4 && isReconstructScale)
	{
		std::vector<cv::Mat> planes;
		cv::split(floatim, planes);

		Reconstruct(false, true, cancel_func, planes[3], alpha);
	}

	// アルファチャンネルがあったらアルファを付加する
	if (!alpha.empty())
	{
		std::vector<cv::Mat> planes;
		cv::split(process_image, planes);
		process_image.release();

		planes.push_back(alpha);

		cv::merge(planes, process_image);
	}

	if (isReconstructScale)
	{
		const cv::Size_<int> ns(image_size.width * shrinkRatio, image_size.height * shrinkRatio);
		if (image_size.width != ns.width || image_size.height != ns.height)
			cv::resize(process_image, process_image, ns, 0.0, 0.0, cv::INTER_LINEAR);
	}

	out = process_image;

	return eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::waifu2x(const boost::filesystem::path &input_file, const boost::filesystem::path &output_file,
	const waifu2xCancelFunc cancel_func)
{
	Waifu2x::eWaifu2xError ret;

	if (!is_inited)
		return eWaifu2xError_NotInitialized;

	const boost::filesystem::path ip(input_file);
	const boost::filesystem::path ipext(ip.extension());

	const bool isJpeg = boost::iequals(ipext.string(), ".jpg") || boost::iequals(ipext.string(), ".jpeg");
	const bool isReconstructNoise = mode == "noise" || mode == "noise_scale" || (mode == "auto_scale" && isJpeg);
	const bool isReconstructScale = mode == "scale" || mode == "noise_scale";

	cv::Mat float_image;
	ret = LoadMat(float_image, input_file);
	if (ret != eWaifu2xError_OK)
		return ret;

	cv::Mat reconstruct_image;
	ret = Reconstruct(isReconstructNoise, isReconstructScale, cancel_func, float_image, reconstruct_image);
	if (ret != eWaifu2xError_OK)
		return ret;

	cv::Mat process_image;
	ret = AfterReconstructFloatMatProcess(isReconstructScale, cancel_func, float_image, reconstruct_image, process_image);
	if (ret != eWaifu2xError_OK)
		return ret;

	float_image.release();

	cv::Mat write_iamge;
	process_image.convertTo(write_iamge, CV_8U, 255.0, clip_eps8);
	process_image.release();

	// 完全透明のピクセルの色を消す(処理の都合上、完全透明のピクセルにも色を付けたから)
	// モデルによっては画像全域の完全透明の場所にごく小さい値のアルファが広がることがある。それを消すためにuint8_tに変換してからこの処理を行うことにした
	if (write_iamge.channels() > 3)
	{
		std::vector<cv::Mat> planes;
		cv::split(write_iamge, planes);

		cv::Mat mask;
		cv::threshold(planes[3], mask, 0.0, 1.0, cv::THRESH_BINARY); // アルファチャンネルを二値化してマスクとして扱う

		// アルファチャンネルが0のところの色を消す
		planes[0] = planes[0].mul(mask);
		planes[1] = planes[1].mul(mask);
		planes[2] = planes[2].mul(mask);

		cv::merge(planes, write_iamge);
	}

	ret = WriteMat(write_iamge, output_file);
	if (ret != eWaifu2xError_OK)
		return ret;

	write_iamge.release();

	return eWaifu2xError_OK;
}

const std::string& Waifu2x::used_process() const
{
	return process;
}
