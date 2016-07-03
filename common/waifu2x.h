#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <utility>
#include <functional>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <opencv2/core.hpp>

#define CUDNN_DLL_NAME "cudnn64_5.dll"
#define CUDNN_REQUIRE_VERION_TEXT "v5 RC"


namespace caffe
{
	template <typename Dtype>
	class Net;
	class NetParameter;
};

class cNet;
class stImage;


class Waifu2x
{
public:
	enum eWaifu2xModelType
	{
		eWaifu2xModelTypeNoise,
		eWaifu2xModelTypeScale,
		eWaifu2xModelTypeNoiseScale,
		eWaifu2xModelTypeAutoScale,
	};

	enum eWaifu2xError
	{
		eWaifu2xError_OK = 0,
		eWaifu2xError_Cancel,
		eWaifu2xError_NotInitialized,
		eWaifu2xError_InvalidParameter,
		eWaifu2xError_FailedOpenInputFile,
		eWaifu2xError_FailedOpenOutputFile,
		eWaifu2xError_FailedOpenModelFile,
		eWaifu2xError_FailedParseModelFile,
		eWaifu2xError_FailedWriteModelFile,
		eWaifu2xError_FailedConstructModel,
		eWaifu2xError_FailedProcessCaffe,
		eWaifu2xError_FailedCudaCheck,
		eWaifu2xError_FailedUnknownType,
	};

	enum eWaifu2xCudaError
	{
		eWaifu2xCudaError_OK = 0,
		eWaifu2xCudaError_NotFind,
		eWaifu2xCudaError_OldVersion,
		eWaifu2xCudaError_OldDevice,
	};

	enum eWaifu2xcuDNNError
	{
		eWaifu2xcuDNNError_OK = 0,
		eWaifu2xcuDNNError_NotFind,
		eWaifu2xcuDNNError_OldVersion,
		eWaifu2xcuDNNError_CannotCreate,
	};

	typedef std::function<bool()> waifu2xCancelFunc;

	static std::string ExeDir;

private:
	bool mIsInited;

	eWaifu2xModelType mMode;
	int mNoiseLevel;
	std::string mProcess;

	bool mIsCuda;

	std::shared_ptr<cNet> mNoiseNet;
	std::shared_ptr<cNet> mScaleNet;

	int mInputPlane; // ネットへの入力チャンネル数
	int mMaxNetOffset; // ネットに入力するとどれくらい削れるか
	bool mHasNoiseScale;

	float *mOutputBlock;
	size_t mOutputBlockSize;

private:
	static boost::filesystem::path GetModeDirPath(const boost::filesystem::path &model_dir);
	static boost::filesystem::path GetInfoPath(const boost::filesystem::path &model_dir);

	Waifu2x::eWaifu2xError ReconstructImage(const double factor, const int crop_w, const int crop_h, const bool use_tta, const int batch_size, 
		const bool isReconstructNoise, const bool isReconstructScale, const Waifu2x::waifu2xCancelFunc cancel_func, stImage &image);
	Waifu2x::eWaifu2xError ReconstructScale(const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
		const Waifu2x::waifu2xCancelFunc cancel_func, stImage &image);
	Waifu2x::eWaifu2xError ReconstructNoiseScale(const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
		const Waifu2x::waifu2xCancelFunc cancel_func, stImage &image);
	Waifu2x::eWaifu2xError ReconstructByNet(std::shared_ptr<cNet> net, const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
		const Waifu2x::waifu2xCancelFunc cancel_func, cv::Mat &im);
	Waifu2x::eWaifu2xError ProcessNet(std::shared_ptr<cNet> net, const int crop_w, const int crop_h, const bool use_tta, const int batch_size, cv::Mat &im);

	// double CalcScaleRatio(const cv::Size_<int> &size) const;

public:
	Waifu2x();
	~Waifu2x();

	static eWaifu2xCudaError can_use_CUDA();
	static eWaifu2xcuDNNError can_use_cuDNN();

	static void init_liblary(int argc, char** argv);
	static void quit_liblary();

	// mode: noise or scale or noise_scale or auto_scale
	// process: cpu or gpu or cudnn
	eWaifu2xError Init(const eWaifu2xModelType mode, const int noise_level,
		const boost::filesystem::path &model_dir, const std::string &process);

	eWaifu2xError waifu2x(const boost::filesystem::path &input_file, const boost::filesystem::path &output_file,
		const double factor, const waifu2xCancelFunc cancel_func = nullptr, const int crop_w = 128, const int crop_h = 128,
		const boost::optional<int> output_quality = boost::optional<int>(), const int output_depth = 8, const bool use_tta = false,
		const int batch_size = 1);

	// factor: 倍率
	// source: (4チャンネルの場合は)RGBAな画素配列
	// dest: (4チャンネルの場合は)処理したRGBAな画素配列
	// in_stride: sourceのストライド(バイト単位)
	// out_stride: destのストライド(バイト単位)
	eWaifu2xError waifu2x(const double factor, const void* source, void* dest, const int width, const int height,
		const int in_channel, const int in_stride, const int out_channel, const int out_stride,
		const int crop_w = 128, const int crop_h = 128,  const bool use_tta = false, const int batch_size = 1);

	void Destroy();

	const std::string& used_process() const;

	static std::string GetModelName(const boost::filesystem::path &model_dir);
};
