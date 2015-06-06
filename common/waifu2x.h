#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <utility>
#include <functional>
#include <memory>
#include <thread>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>


namespace caffe
{
	template <typename Dtype>
	class Net;
};

namespace tinypl
{
	namespace impl
	{
		// task scheduler
		class scheduler;
	}
}

class Waifu2x
{
public:
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
		eWaifu2xError_FailedConstructModel,
		eWaifu2xError_FailedProcessCaffe,
	};

	enum eWaifu2xcuDNNError
	{
		eWaifu2xcuDNNError_OK = 0,
		eWaifu2xcuDNNError_NotFind,
		eWaifu2xcuDNNError_OldVersion,
		eWaifu2xcuDNNError_CannotCreate,
	};

	typedef std::function<bool()> waifu2xCancelFunc;

private:
	bool is_inited;

	// 一度に処理する画像の幅
	int crop_size;
	// 一度に何ブロック分処理するか
	int batch_size;

	// ネットに入力する画像のサイズ
	int block_size;
	// ブロック変換後の出力サイズ
	int output_size;
	// ネットワークに入力する画像のサイズ(出力画像の幅はlayer_num * 2だけ小さくなる)
	int block_width_height;
	// srcnn.prototxtで定義された入力する画像のサイズ
	int original_width_height;

	std::string mode;
	int noise_level;
	double scale_ratio;
	std::string model_dir;
	std::string process;
	int job;

	bool isCuda;

	std::vector<boost::shared_ptr<caffe::Net<float>>> net_noises;
	std::vector<boost::shared_ptr<caffe::Net<float>>> net_scales;

	std::vector<float *> blocks;
	std::vector<float *> dummy_datas;
	std::vector<float *> out_blocks;

	std::unique_ptr<tinypl::impl::scheduler> net_scheduler;
	std::unordered_map<std::thread::id, size_t> net_scheduler_id_map;

private:
	eWaifu2xError LoadImage(cv::Mat &float_image, const std::string &input_file);
	eWaifu2xError CreateBrightnessImage(const cv::Mat &float_image, cv::Mat &im);
	eWaifu2xError PaddingImage(const cv::Mat &input, cv::Mat &output);
	eWaifu2xError Zoom2xAndPaddingImage(const cv::Mat &input, cv::Mat &output, cv::Size_<int> &zoom_size);
	eWaifu2xError CreateZoomColorImage(const cv::Mat &float_image, const cv::Size_<int> &zoom_size, std::vector<cv::Mat> &cubic_planes);
	eWaifu2xError LoadParameter(boost::shared_ptr<caffe::Net<float>> net, const std::string &param_path);
	eWaifu2xError ConstractNet(boost::shared_ptr<caffe::Net<float>> &net, const std::string &model_path, const std::string &process);
	eWaifu2xError ReconstructImage(std::vector<boost::shared_ptr<caffe::Net<float>>> nets, cv::Mat &im);

public:
	Waifu2x();
	~Waifu2x();

	static eWaifu2xcuDNNError can_use_cuDNN();

	// mode: noise or scale or noise_scale or auto_scale
	// process: cpu or gpu or cudnn
	eWaifu2xError init(int argc, char** argv, const std::string &mode, const int noise_level, const double scale_ratio, const std::string &model_dir, const std::string &process,
		const int crop_size = 128, const int batch_size = 1);

	void destroy();

	eWaifu2xError waifu2x(const std::string &input_file, const std::string &output_file,
		const waifu2xCancelFunc cancel_func = nullptr);

	const std::string& used_process() const;
};
