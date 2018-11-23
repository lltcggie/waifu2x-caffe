#include "waifu2x.h"
#include "stImage.h"
#include "cNet.h"
#include <caffe/caffe.hpp>
#include <cudnn.h>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <tclap/CmdLine.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <unordered_map>
#include <cuda_runtime.h>

#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <msgpack.hpp>

#include <fcntl.h>
#include <zlib.h>
#ifdef _MSC_VER
#include <io.h>
#endif

//#if defined(WIN32) || defined(WIN64)
//#include <Windows.h>
//#endif

#define CV_VERSION_STR CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

// ビルドモード
#ifdef _DEBUG
#define CV_EXT_STR "d.lib"
#else
#define CV_EXT_STR ".lib"
#endif

#ifdef _MSC_VER

#pragma comment(lib, "opencv_core" CV_VERSION_STR CV_EXT_STR)
#pragma comment(lib, "opencv_imgcodecs" CV_VERSION_STR CV_EXT_STR)
#pragma comment(lib, "opencv_imgproc" CV_VERSION_STR CV_EXT_STR)
#pragma comment(lib, "IlmImf" CV_EXT_STR)
#pragma comment(lib, "ippicvmt.lib")
#pragma comment(lib, "libjasper" CV_EXT_STR)
#pragma comment(lib, "libjpeg-turbo" CV_EXT_STR)
#pragma comment(lib, "libpng" CV_EXT_STR)
#pragma comment(lib, "libtiff" CV_EXT_STR)
#pragma comment(lib, "libwebp" CV_EXT_STR)
#pragma comment(lib, "zlib" CV_EXT_STR)

#pragma comment(lib, "libopenblas.dll.a")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

#ifdef _DEBUG
#pragma comment(lib, "caffe-d.lib")
#pragma comment(lib, "caffeproto-d.lib")
#pragma comment(lib, "libprotobufd.lib")
#pragma comment(lib, "glogd.lib")
#pragma comment(lib, "gflagsd.lib")
#pragma comment(lib, "libboost_system-vc140-mt-gd-1_61.lib")
#pragma comment(lib, "boost_thread-vc140-mt-gd-1_61.lib")
#pragma comment(lib, "boost_filesystem-vc140-mt-gd-1_61.lib")
#pragma comment(lib, "boost_iostreams-vc140-mt-gd-1_61.lib")
//#pragma comment(lib, "zlibstaticd.lib")


#else
#pragma comment(lib, "caffe.lib")
#pragma comment(lib, "caffeproto.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "glog.lib")
#pragma comment(lib, "gflags.lib")
#pragma comment(lib, "libboost_system-vc140-mt-1_61.lib")
#pragma comment(lib, "boost_thread-vc140-mt-1_61.lib")
#pragma comment(lib, "boost_filesystem-vc140-mt-1_61.lib")
#pragma comment(lib, "boost_iostreams-vc140-mt-1_61.lib")
#endif
#endif

const int ScaleBase = 2; // TODO: モデルの拡大率によって可変できるようにする

// 入力画像に追加するパディング
const int OuterPadding = 0;

// 最低限必要なCUDAドライバーのバージョン
const int MinCudaDriverVersion = 7050;

static std::once_flag waifu2x_once_flag;
static std::once_flag waifu2x_cudnn_once_flag;
static std::once_flag waifu2x_cuda_once_flag;

std::string Waifu2x::ExeDir;


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

	class CudaDeviceSet
	{
	private:
		int orgDevice;
		bool mIsSet;

	public:
		CudaDeviceSet(const std::string &process, const int devno) : orgDevice(0), mIsSet(false)
		{
			if (process == "gpu" || process == "cudnn")
			{
				int count = 0;
				if (cudaGetDeviceCount(&count) != CUDA_SUCCESS)
					return;

				if (devno >= count || count < 0)
					return;

				if (cudaGetDevice(&orgDevice) != CUDA_SUCCESS)
					return;

				if (cudaSetDevice(devno) != CUDA_SUCCESS)
					return;

				mIsSet = true;
			}
		}

		~CudaDeviceSet()
		{
			if (mIsSet)
				cudaSetDevice(orgDevice);
		}
	};
}

class CcuDNNAlgorithmElement
{
private:
	typedef std::unordered_map<uint64_t, uint8_t> AlgoMap;

	AlgoMap mAlgo;
	bool mIsModefy;

	uint8_t kernel_w;
	uint8_t kernel_h;
	uint8_t pad_w;
	uint8_t pad_h;
	uint8_t stride_w;
	uint8_t stride_h;
	uint16_t batch_size;

private:
	static uint64_t InfoToKey(uint16_t num_input, uint16_t num_output, uint16_t width, uint16_t height)
	{
		return (uint64_t)num_input << 8 * 3 | (uint64_t)num_output << 8 * 2 | (uint64_t)width << 8 * 1 | (uint64_t)height << 8 * 0;
	}

public:
	CcuDNNAlgorithmElement() : mIsModefy(false)
	{}
	~CcuDNNAlgorithmElement()
	{}

	void SetLayerData(uint8_t kernel_w, uint8_t kernel_h, uint8_t pad_w, uint8_t pad_h, uint8_t stride_w, uint8_t stride_h, uint16_t batch_size)
	{
		this->kernel_w = kernel_w;
		this->kernel_h = kernel_h;
		this->pad_w = pad_w;
		this->pad_h = pad_h;
		this->stride_w = stride_w;
		this->stride_h = stride_h;
		this->batch_size = batch_size;
	}

	void GetLayerData(uint8_t &kernel_w, uint8_t &kernel_h, uint8_t &pad_w, uint8_t &pad_h, uint8_t &stride_w, uint8_t &stride_h, uint16_t &batch_size)
	{
		kernel_w = this->kernel_w;
		kernel_h = this->kernel_h;
		pad_w = this->pad_w;
		pad_h = this->pad_h;
		stride_w = this->stride_w;
		stride_h = this->stride_h;
		batch_size = this->batch_size;
	}

	int GetAlgorithm(uint16_t num_input, uint16_t num_output, uint16_t width, uint16_t height) const
	{
		const uint64_t key = InfoToKey(num_input, num_output, width, height);
		const auto it = mAlgo.find(key);
		if (it != mAlgo.end())
			return it->second;

		return -1;
	}

	void SetAlgorithm(uint8_t algo, uint16_t num_input, uint16_t num_output, uint16_t width, uint16_t height)
	{
		const uint64_t key = InfoToKey(num_input, num_output, width, height);
		auto it = mAlgo.find(key);
		if (it == mAlgo.end() || it->second != algo)
			mIsModefy = true;

		mAlgo[key] = algo;
	}

	bool IsModefy() const
	{
		return mIsModefy;
	}

	void Saved()
	{
		mIsModefy = false;
	}

	MSGPACK_DEFINE(mAlgo, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, batch_size);
};

class CcuDNNAlgorithm
{
private:
	typedef std::unordered_map<uint64_t, CcuDNNAlgorithmElement> AlgoEmlMap;

	AlgoEmlMap mAlgoEmlMap;
	std::string mDataPath;

private:
	static uint64_t InfoToKey(uint8_t kernel_w, uint8_t kernel_h, uint8_t pad_w, uint8_t pad_h, uint8_t stride_w, uint8_t stride_h, uint16_t batch_size)
	{
		return
			(uint64_t)kernel_w << 8 * 7 | (uint64_t)kernel_h << 8 * 6 |
			(uint64_t)pad_w << 8 * 5 | (uint64_t)pad_h << 8 * 4 |
			(uint64_t)stride_w << 8 * 3 | (uint64_t)stride_h << 8 * 2 |
			(uint64_t)batch_size;
	}

	std::string GetDataPath(uint8_t kernel_w, uint8_t kernel_h, uint8_t pad_w, uint8_t pad_h, uint8_t stride_w, uint8_t stride_h, uint16_t batch_size) const
	{
		std::string SavePath = mDataPath;
		SavePath +=
			std::to_string(kernel_w) + "x" + std::to_string(kernel_h) + " " +
			std::to_string(pad_w) + "x" + std::to_string(pad_w) + " " +
			std::to_string(stride_w) + "x" + std::to_string(stride_w) + " " +
			std::to_string(batch_size);
		SavePath += ".dat";

		return SavePath;
	}

	bool Load(uint8_t kernel_w, uint8_t kernel_h, uint8_t pad_w, uint8_t pad_h, uint8_t stride_w, uint8_t stride_h, uint16_t batch_size)
	{
		const std::string SavePath = GetDataPath(kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, batch_size);

		std::vector<char> sbuf;

		FILE *fp = fopen(SavePath.c_str(), "rb");
		if (!fp)
			return false;

		fseek(fp, 0, SEEK_END);
		const auto size = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		sbuf.resize(size);

		if (fread(sbuf.data(), 1, sbuf.size(), fp) != sbuf.size())
		{
			fclose(fp);
			return false;
		}

		fclose(fp);

		try
		{
			CcuDNNAlgorithmElement elm;
			msgpack::unpack(sbuf.data(), sbuf.size()).get().convert(elm);
			sbuf.clear();

			const uint64_t key = InfoToKey(kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, batch_size);
			mAlgoEmlMap[key] = std::move(elm);
		}
		catch (...)
		{
			boost::filesystem::remove(SavePath);
		}

		return true;
	}

public:
	CcuDNNAlgorithm()
	{}

	~CcuDNNAlgorithm()
	{
		Save();
	}

	int GetAlgorithm(uint16_t num_input, uint16_t num_output, uint16_t batch_size,
		uint16_t width, uint16_t height, uint16_t kernel_w, uint16_t kernel_h, uint16_t pad_w, uint16_t pad_h, uint16_t stride_w, uint16_t stride_h)
	{
		const uint64_t key = InfoToKey(kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, batch_size);
		const auto it = mAlgoEmlMap.find(key);
		if (it != mAlgoEmlMap.end())
		{
			const auto &elm = it->second;
			return elm.GetAlgorithm(num_input, num_output, width, height);
		}

		if (Load(kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, batch_size))
			return mAlgoEmlMap[key].GetAlgorithm(num_input, num_output, width, height);

		return -1;
	}

	void SetAlgorithm(int algo, uint16_t num_input, uint16_t num_output, uint16_t batch_size,
		uint16_t width, uint16_t height, uint16_t kernel_w, uint16_t kernel_h, uint16_t pad_w, uint16_t pad_h, uint16_t stride_w, uint16_t stride_h)
	{
		if (algo < 0 || algo > 255)
			return;

		const uint64_t key = InfoToKey(kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, batch_size);
		auto &eml = mAlgoEmlMap[key];
		eml.SetAlgorithm(algo, num_input, num_output, width, height);
		eml.SetLayerData(kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, batch_size);
	}

	void Save()
	{
		for (auto &p : mAlgoEmlMap)
		{
			auto &eml = p.second;
			if (eml.IsModefy())
			{
				try
				{
					msgpack::sbuffer sbuf;
					msgpack::pack(sbuf, eml);

					uint8_t kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h;
					uint16_t batch_size;
					eml.GetLayerData(kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, batch_size);

					const std::string SavePath = GetDataPath(kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, batch_size);
					FILE *fp = fopen(SavePath.c_str(), "wb");
					if (fp)
					{
						fwrite(sbuf.data(), 1, sbuf.size(), fp);
						fclose(fp);

						eml.Saved();
					}
				}
				catch(...)
				{}
			}
		}
	}

	void SetDataPath(std::string path)
	{
		mDataPath = path;
	}
};

CcuDNNAlgorithm g_ConvCcuDNNAlgorithm;
CcuDNNAlgorithm g_DeconvCcuDNNAlgorithm;


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
					{
						cudaDeviceProp prop;
						cudaGetDeviceProperties(&prop, 0);
						if (prop.major >= 2)
							CudaFlag = eWaifu2xCudaError_OK;
						else
							CudaFlag = eWaifu2xCudaError_OldDevice;
					}
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
				if (cudnnGetVersionFunc() >= CUDNN_REQUIRE_VERION)
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

void Waifu2x::init_liblary(int argc, char** argv)
{
	if (argc > 0)
		ExeDir = argv[0];

	std::call_once(waifu2x_once_flag, [argc, argv]()
	{
		assert(argc >= 1);

		int tmpargc = 1;
		char* tmpargvv[] = {argv[0]};
		char** tmpargv = tmpargvv;
		// glog等の初期化
		caffe::GlobalInit(&tmpargc, &tmpargv);
	});
}

void Waifu2x::quit_liblary()
{
	g_ConvCcuDNNAlgorithm.Save();
	g_DeconvCcuDNNAlgorithm.Save();

	//caffe::GlobalFinalize();
}

void Waifu2x::quit_thread_liblary()
{
	//caffe::ThreadFinalize();
}

Waifu2x::Waifu2x() : mIsInited(false), mNoiseLevel(0), mIsCuda(false), mOutputBlock(nullptr), mOutputBlockSize(0), mGPUNo(0)
{}

Waifu2x::~Waifu2x()
{
	Destroy();
}

Waifu2x::eWaifu2xError Waifu2x::Init(const eWaifu2xModelType mode, const int noise_level,
	const boost::filesystem::path &model_dir, const std::string &process, const int GPUNo)
{
	Waifu2x::eWaifu2xError ret;

	if (mIsInited)
		return Waifu2x::eWaifu2xError_OK;

	try
	{
		std::string Process = process;
		const auto cuDNNCheckStartTime = std::chrono::system_clock::now();

		if (Process == "gpu")
		{
			if (can_use_CUDA() != eWaifu2xCudaError_OK)
				return Waifu2x::eWaifu2xError_FailedCudaCheck;
			// cuDNNが使えそうならcuDNNを使う
			else if (can_use_cuDNN() == eWaifu2xcuDNNError_OK)
				Process = "cudnn";
		}

		mMode = mode;
		mNoiseLevel = noise_level;
		mProcess = Process;
		mGPUNo = GPUNo;

		const auto cuDNNCheckEndTime = std::chrono::system_clock::now();

		if (Process == "cudnn")
		{
			// exeのディレクトリにcuDNNのアルゴリズムデータ保存
			boost::filesystem::path cudnn_data_base_dir_path(ExeDir);
			if (cudnn_data_base_dir_path.is_relative())
				cudnn_data_base_dir_path = boost::filesystem::system_complete(cudnn_data_base_dir_path);

			if (!boost::filesystem::is_directory(cudnn_data_base_dir_path))
				cudnn_data_base_dir_path = cudnn_data_base_dir_path.branch_path();

			if (!boost::filesystem::exists(cudnn_data_base_dir_path))
			{
				// exeのディレクトリが取得できなければカレントディレクトリに保存

				cudnn_data_base_dir_path = boost::filesystem::current_path();

				if (cudnn_data_base_dir_path.is_relative())
					cudnn_data_base_dir_path = boost::filesystem::system_complete(cudnn_data_base_dir_path);

				if (!boost::filesystem::exists(cudnn_data_base_dir_path))
					cudnn_data_base_dir_path = "./";
			}

			if (boost::filesystem::exists(cudnn_data_base_dir_path))
			{
				const boost::filesystem::path cudnn_data_dir_path(cudnn_data_base_dir_path / "cudnn_data");

				bool isOK = false;
				if (boost::filesystem::exists(cudnn_data_dir_path))
					isOK = true;

				if (!isOK)
				{
					boost::system::error_code error;
					const bool result = boost::filesystem::create_directory(cudnn_data_dir_path, error);
					if (result && !error)
						isOK = true;
				}

				if (isOK)
				{
					cudaDeviceProp prop;
					if (cudaGetDeviceProperties(&prop, mGPUNo) == cudaSuccess)
					{
						std::string conv_filename(prop.name);
						conv_filename += " conv ";

						std::string deconv_filename(prop.name);
						deconv_filename += " deconv ";

						const boost::filesystem::path conv_data_path = cudnn_data_dir_path / conv_filename;
						const boost::filesystem::path deconv_data_path = cudnn_data_dir_path / deconv_filename;

						g_ConvCcuDNNAlgorithm.SetDataPath(conv_data_path.string());
						g_DeconvCcuDNNAlgorithm.SetDataPath(deconv_data_path.string());
					}
				}
			}
		}

		const boost::filesystem::path mode_dir_path(GetModeDirPath(model_dir));
		if (!boost::filesystem::exists(mode_dir_path))
			return Waifu2x::eWaifu2xError_FailedOpenModelFile;

		CudaDeviceSet devset(process, mGPUNo);

		if (mProcess == "cpu")
		{
			caffe::Caffe::set_mode(caffe::Caffe::CPU);
			mIsCuda = false;
		}
		else
		{
			caffe::Caffe::set_mode(caffe::Caffe::GPU);
			mIsCuda = true;
		}

		caffe::Caffe::SetGetcuDNNAlgorithmFunc(GetcuDNNAlgorithm);
		caffe::Caffe::SetSetcuDNNAlgorithmFunc(SetcuDNNAlgorithm);

		mInputPlane = 0;
		mMaxNetOffset = 0;

		const boost::filesystem::path info_path = GetInfoPath(mode_dir_path);

		stInfo info;
		ret = cNet::GetInfo(info_path, info);
		if (ret != Waifu2x::eWaifu2xError_OK)
			return ret;

		mHasNoiseScale = info.has_noise_scale;
		mInputPlane = info.channels;

		if (mode == eWaifu2xModelTypeNoise || mode == eWaifu2xModelTypeNoiseScale || mode == eWaifu2xModelTypeAutoScale)
		{
			std::string base_name;

			mNoiseNet.reset(new cNet);

			eWaifu2xModelType Mode = mode;
			if (info.has_noise_scale) // ノイズ除去と拡大を同時に行う
			{
				// ノイズ除去拡大ネットの構築はeWaifu2xModelTypeNoiseScaleを指定する必要がある
				Mode = eWaifu2xModelTypeNoiseScale;
				base_name = "noise" + std::to_string(noise_level) + "_scale2.0x_model";
			}
			else // ノイズ除去だけ
			{
				Mode = eWaifu2xModelTypeNoise;
				base_name = "noise" + std::to_string(noise_level) + "_model";
			}

			const boost::filesystem::path model_path = mode_dir_path / (base_name + ".prototxt");
			const boost::filesystem::path param_path = mode_dir_path / (base_name + ".json");

			ret = mNoiseNet->ConstractNet(Mode, model_path, param_path, info, mProcess);
			if (ret != Waifu2x::eWaifu2xError_OK)
				return ret;

			mMaxNetOffset = mNoiseNet->GetNetOffset();
		}

		// noise_scaleを持っている場合はαチャンネルの拡大のためにmScaleNetも構築する必要がある
		if (info.has_noise_scale || mode == eWaifu2xModelTypeScale || mode == eWaifu2xModelTypeNoiseScale || mode == eWaifu2xModelTypeAutoScale)
		{
			const std::string base_name = "scale2.0x_model";

			const boost::filesystem::path model_path = mode_dir_path / (base_name + ".prototxt");
			const boost::filesystem::path param_path = mode_dir_path / (base_name + ".json");

			mScaleNet.reset(new cNet);

			ret = mScaleNet->ConstractNet(eWaifu2xModelTypeScale, model_path, param_path, info, mProcess);
			if (ret != Waifu2x::eWaifu2xError_OK)
				return ret;

			assert(mInputPlane == 0 || mInputPlane == mScaleNet->GetInputPlane());

			mMaxNetOffset = std::max(mScaleNet->GetNetOffset(), mMaxNetOffset);
		}
		else
		{
			
		}

		mIsInited = true;
	}
	catch (...)
	{
		return Waifu2x::eWaifu2xError_InvalidParameter;
	}

	return Waifu2x::eWaifu2xError_OK;
}

boost::filesystem::path Waifu2x::GetModeDirPath(const boost::filesystem::path &model_dir)
{
	boost::filesystem::path mode_dir_path(model_dir);
	if (!mode_dir_path.is_absolute()) // model_dirが相対パスなら絶対パスに直す
	{
		// まずはカレントディレクトリ下にあるか探す
		mode_dir_path = boost::filesystem::absolute(model_dir);
		if (!boost::filesystem::exists(mode_dir_path) && !ExeDir.empty()) // 無かったらargv[0]から実行ファイルのあるフォルダを推定し、そのフォルダ下にあるか探す
		{
			boost::filesystem::path a0(ExeDir);
			if (a0.is_absolute())
				mode_dir_path = a0.branch_path() / model_dir;
		}
	}

	return mode_dir_path;
}

boost::filesystem::path Waifu2x::GetInfoPath(const boost::filesystem::path &mode_dir_path)
{
	const boost::filesystem::path info_path = mode_dir_path / "info.json";

	return info_path;
}

Waifu2x::eWaifu2xError Waifu2x::waifu2x(const boost::filesystem::path &input_file, const boost::filesystem::path &output_file,
	const boost::optional<double> scale_ratio, const boost::optional<int> scale_width, const boost::optional<int> scale_height, 
	const waifu2xCancelFunc cancel_func, const int crop_w, const int crop_h,
	const boost::optional<int> output_quality, const int output_depth, const bool use_tta,
	const int batch_size)
{
	Waifu2x::eWaifu2xError ret;

	if (!mIsInited)
		return Waifu2x::eWaifu2xError_NotInitialized;

	stImage image;
	ret = image.Load(input_file);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	image.Preprocess(mInputPlane, mMaxNetOffset);

	const bool isReconstructNoise = mMode == eWaifu2xModelTypeNoise || mMode == eWaifu2xModelTypeNoiseScale || (mMode == eWaifu2xModelTypeAutoScale && image.RequestDenoise());
	const bool isReconstructScale = mMode == eWaifu2xModelTypeScale || mMode == eWaifu2xModelTypeNoiseScale || mMode == eWaifu2xModelTypeAutoScale;

	auto factor = CalcScaleRatio(scale_ratio, scale_width, scale_height, image);

	if (!isReconstructScale)
		factor = Factor(1.0, 1.0);

	cv::Mat reconstruct_image;
	ret = ReconstructImage(factor, crop_w, crop_h, use_tta, batch_size, isReconstructNoise, isReconstructScale, cancel_func, image);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	if(!scale_width || !scale_height)
		image.Postprocess(mInputPlane, factor, output_depth);
	else
		image.Postprocess(mInputPlane, *scale_width, *scale_height, output_depth);

	ret = image.Save(output_file, output_quality);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	return Waifu2x::eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::waifu2x(const double factor, const void* source, void* dest, const int width, const int height,
	const int in_channel, const int in_stride, const int out_channel, const int out_stride,
	const int crop_w, const int crop_h, const bool use_tta, const int batch_size)
{
	Waifu2x::eWaifu2xError ret;

	if (!mIsInited)
		return Waifu2x::eWaifu2xError_NotInitialized;

	int cvrSetting = -1;
	if (in_channel == 3 && out_channel == 3)
		cvrSetting = CV_BGR2RGB;
	else if (in_channel == 4 && out_channel == 4)
		cvrSetting = CV_BGRA2RGBA;
	else if (in_channel == 3 && out_channel == 4)
		cvrSetting = CV_BGR2RGBA;
	else if (in_channel == 4 && out_channel == 3)
		cvrSetting = CV_BGRA2RGB;
	else if (!(in_channel == 1 && out_channel == 1))
		return Waifu2x::eWaifu2xError_InvalidParameter;

	stImage image;
	ret = image.Load(source, width, height, in_channel, in_stride);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	image.Preprocess(mInputPlane, mMaxNetOffset);

	const bool isReconstructNoise = mMode == eWaifu2xModelTypeNoise || mMode == eWaifu2xModelTypeNoiseScale;
	const bool isReconstructScale = mMode == eWaifu2xModelTypeScale || mMode == eWaifu2xModelTypeNoiseScale || mMode == eWaifu2xModelTypeAutoScale;

	Factor nowFactor = Factor(factor, 1.0);
	if (!isReconstructScale)
		nowFactor = Factor(1.0, 1.0);

	cv::Mat reconstruct_image;
	ret = ReconstructImage(nowFactor, crop_w, crop_h, use_tta, batch_size, isReconstructNoise, isReconstructScale, nullptr, image);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	image.Postprocess(mInputPlane, nowFactor, 8);

	cv::Mat out_bgr_image = image.GetEndImage();
	image.Clear();

	cv::Mat out_image;
	if (cvrSetting >= 0)
		cv::cvtColor(out_bgr_image, out_image, cvrSetting); // BGRからRGBに戻す
	else
		out_image = out_bgr_image;
	out_bgr_image.release();

	// 出力配列へ書き込み
	{
		const auto width = out_image.size().width;
		const auto stride = out_image.step1();
		for (int i = 0; i < out_image.size().height; i++)
			memcpy((uint8_t *)dest + out_stride * i, out_image.data + stride * i, out_stride);
	}

	return Waifu2x::eWaifu2xError_OK;
}

Factor Waifu2x::CalcScaleRatio(const boost::optional<double> scale_ratio, const boost::optional<int> scale_width, const boost::optional<int> scale_height,
	const stImage &image)
{
	if (scale_ratio)
		return Factor(*scale_ratio, 1.0);

	if (scale_width && scale_height)
	{
		const auto d1 = image.GetScaleFromWidth(*scale_width);
		const auto d2 = image.GetScaleFromWidth(*scale_height);

		return d1.toDouble() >= d2.toDouble() ? d1 : d2;
	}

	if (scale_width)
		return image.GetScaleFromWidth(*scale_width);

	if(scale_height)
		return image.GetScaleFromHeight(*scale_height);

	return Factor(1.0, 1.0);
}

int Waifu2x::GetcuDNNAlgorithm(const char * layer_name, int num_input, int num_output, int batch_size,
	int width, int height, int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h)
{
	// g_ConvCcuDNNAlgorithmとg_DeconvCcuDNNAlgorithmが逆になってしまっているが、ファイル名にしか影響がないのと互換性がなくなるのでこのまま仕様とする
	if (strcmp(layer_name, "Deconvolution") == 0)
		return g_ConvCcuDNNAlgorithm.GetAlgorithm(num_input, num_output, batch_size, width, height, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
	else if (strcmp(layer_name, "Convolution") == 0)
		return g_DeconvCcuDNNAlgorithm.GetAlgorithm(num_input, num_output, batch_size, width, height, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);

	return -1;
}

void Waifu2x::SetcuDNNAlgorithm(int algo, const char * layer_name, int num_input, int num_output, int batch_size,
	int width, int height, int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h)
{
	// g_ConvCcuDNNAlgorithmとg_DeconvCcuDNNAlgorithmが逆になってしまっているが、ファイル名にしか影響がないのと互換性がなくなるのでこのまま仕様とする
	if (strcmp(layer_name, "Deconvolution") == 0)
		return g_ConvCcuDNNAlgorithm.SetAlgorithm(algo, num_input, num_output, batch_size, width, height, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
	else if (strcmp(layer_name, "Convolution") == 0)
		return g_DeconvCcuDNNAlgorithm.SetAlgorithm(algo, num_input, num_output, batch_size, width, height, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
}

Waifu2x::eWaifu2xError Waifu2x::ReconstructImage(const Factor factor, const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
	const bool isReconstructNoise, const bool isReconstructScale, const Waifu2x::waifu2xCancelFunc cancel_func, stImage &image)
{
	Waifu2x::eWaifu2xError ret;

	Factor nowFactor = factor;

	if (isReconstructNoise)
	{
		if (!mHasNoiseScale) // ノイズ除去だけ
		{
			cv::Mat im;
			cv::Size_<int> size;
			image.GetScalePaddingedRGB(im, size, mNoiseNet->GetNetOffset(), OuterPadding, crop_w, crop_h, 1);

			ret = ReconstructByNet(mNoiseNet, crop_w, crop_h, use_tta, batch_size, cancel_func, im);
			if (ret != Waifu2x::eWaifu2xError_OK)
				return ret;

			image.SetReconstructedRGB(im, size, 1);
		}
		else // ノイズ除去と拡大
		{
			ret = ReconstructNoiseScale(crop_w, crop_h, use_tta, batch_size, cancel_func, image);
			if (ret != Waifu2x::eWaifu2xError_OK)
				return ret;

			//nowFactor /= mNoiseNet->GetInnerScale();
			nowFactor = nowFactor.MultiDenominator(mNoiseNet->GetInnerScale());
		}
	}

	if (cancel_func && cancel_func())
		return Waifu2x::eWaifu2xError_Cancel;

	const int scaleNum = ceil(log(nowFactor.toDouble()) / log(ScaleBase));

	if (isReconstructScale)
	{
		for (int i = 0; i < scaleNum; i++)
		{
			ret = ReconstructScale(crop_w, crop_h, use_tta, batch_size, cancel_func, image);
			if (ret != Waifu2x::eWaifu2xError_OK)
				return ret;
		}
	}

	return Waifu2x::eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::ReconstructScale(const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
	const Waifu2x::waifu2xCancelFunc cancel_func, stImage &image)
{
	Waifu2x::eWaifu2xError ret;

	if (image.HasAlpha())
	{
		cv::Mat im;
		cv::Size_<int> size;
		image.GetScalePaddingedA(im, size, mScaleNet->GetNetOffset(), OuterPadding, crop_w, crop_h, mScaleNet->GetScale() / mScaleNet->GetInnerScale());

		ret = ReconstructByNet(mScaleNet, crop_w, crop_h, use_tta, batch_size, cancel_func, im);
		if (ret != Waifu2x::eWaifu2xError_OK)
			return ret;

		image.SetReconstructedA(im, size, mScaleNet->GetInnerScale());
	}

	cv::Mat im;
	cv::Size_<int> size;
	image.GetScalePaddingedRGB(im, size, mScaleNet->GetNetOffset(), OuterPadding, crop_w, crop_h, mScaleNet->GetScale() / mScaleNet->GetInnerScale());

	ret = ReconstructByNet(mScaleNet, crop_w, crop_h, use_tta, batch_size, cancel_func, im);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	image.SetReconstructedRGB(im, size, mScaleNet->GetInnerScale());

	return Waifu2x::eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::ReconstructNoiseScale(const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
	const Waifu2x::waifu2xCancelFunc cancel_func, stImage &image)
{
	Waifu2x::eWaifu2xError ret;

	if (image.HasAlpha())
	{
		// αチャンネルにはノイズ除去を行わない

		cv::Mat im;
		cv::Size_<int> size;
		image.GetScalePaddingedA(im, size, mScaleNet->GetNetOffset(), OuterPadding, crop_w, crop_h, mScaleNet->GetScale() / mScaleNet->GetInnerScale());

		ret = ReconstructByNet(mScaleNet, crop_w, crop_h, use_tta, batch_size, cancel_func, im);
		if (ret != Waifu2x::eWaifu2xError_OK)
			return ret;

		image.SetReconstructedA(im, size, mScaleNet->GetInnerScale());
	}

	cv::Mat im;
	cv::Size_<int> size;
	image.GetScalePaddingedRGB(im, size, mNoiseNet->GetNetOffset(), OuterPadding, crop_w, crop_h, mNoiseNet->GetScale() / mNoiseNet->GetInnerScale());

	ret = ReconstructByNet(mNoiseNet, crop_w, crop_h, use_tta, batch_size, cancel_func, im);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	image.SetReconstructedRGB(im, size, mNoiseNet->GetInnerScale());

	return Waifu2x::eWaifu2xError_OK;
}


Waifu2x::eWaifu2xError Waifu2x::ReconstructByNet(std::shared_ptr<cNet> net, const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
	const Waifu2x::waifu2xCancelFunc cancel_func, cv::Mat &im)
{
	Waifu2x::eWaifu2xError ret;

	if (!use_tta) // 普通に処理
	{
		ret = ProcessNet(net, crop_w, crop_h, use_tta, batch_size, im);
		if (ret != Waifu2x::eWaifu2xError_OK)
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

		cv::Mat reconstruct_image;
		for (int i = 0; i < 8; i++)
		{
			cv::Mat in(im.clone());

			const int rotateNum = i % 4;
			RotateClockwise90N(in, rotateNum);

			if (i >= 4)
				cv::flip(in, in, 1); // 垂直軸反転

			const int cw = (rotateNum % 2 == 0) ? crop_w : crop_h;
			const int ch = (rotateNum % 2 == 0) ? crop_h : crop_w;

			ret = ProcessNet(net, cw, ch, use_tta, batch_size, in);
			if (ret != Waifu2x::eWaifu2xError_OK)
				return ret;

			if (i >= 4)
				cv::flip(in, in, 1); // 垂直軸反転

			RotateCounterclockwise90N(in, rotateNum);

			if (i == 0)
				reconstruct_image = in;
			else
				reconstruct_image += in;
		}

		reconstruct_image /= 8.0;

		im = reconstruct_image;
	}

	return Waifu2x::eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError Waifu2x::ProcessNet(std::shared_ptr<cNet> net, const int crop_w, const int crop_h, const bool use_tta, const int batch_size, cv::Mat &im)
{
	Waifu2x::eWaifu2xError ret;

	CudaDeviceSet devset(mProcess, mGPUNo);

	const auto OutputMemorySize = net->GetOutputMemorySize(crop_w, crop_h, OuterPadding, batch_size);
	if (OutputMemorySize > mOutputBlockSize)
	{
		if (mIsCuda)
		{
			CUDA_HOST_SAFE_FREE(mOutputBlock);
			CUDA_CHECK_WAIFU2X(cudaHostAlloc(&mOutputBlock, OutputMemorySize, cudaHostAllocDefault));
		}
		else
		{
			SAFE_DELETE_WAIFU2X(mOutputBlock);
			mOutputBlock = new float[OutputMemorySize];
		}

		mOutputBlockSize = OutputMemorySize;
	}

	ret = net->ReconstructImage(use_tta, crop_w, crop_h, OuterPadding, batch_size, mOutputBlock, im, im);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	return Waifu2x::eWaifu2xError_OK;
}

void Waifu2x::Destroy()
{
	CudaDeviceSet devset(mProcess, mGPUNo);

	mNoiseNet.reset();
	mScaleNet.reset();

	if (mIsCuda)
	{
		CUDA_HOST_SAFE_FREE(mOutputBlock);
	}
	else
	{
		SAFE_DELETE_WAIFU2X(mOutputBlock);
	}

	mIsInited = false;
}

const std::string& Waifu2x::used_process() const
{
	return mProcess;
}

std::string Waifu2x::GetModelName(const boost::filesystem::path & model_dir)
{
	const boost::filesystem::path mode_dir_path(GetModeDirPath(model_dir));
	if (!boost::filesystem::exists(mode_dir_path))
		return std::string();

	const boost::filesystem::path info_path = mode_dir_path / "info.json";

	return cNet::GetModelName(info_path);
}

bool Waifu2x::GetInfo(const boost::filesystem::path &model_dir, stInfo &info)
{
	const boost::filesystem::path mode_dir_path(GetModeDirPath(model_dir));
	if (!boost::filesystem::exists(mode_dir_path))
		return false;

	const boost::filesystem::path info_path = mode_dir_path / "info.json";

	return cNet::GetInfo(info_path, info) == Waifu2x::eWaifu2xError_OK;
}
