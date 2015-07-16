#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <utility>
#include <functional>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>


namespace caffe
{
	template <typename Dtype>
	class Net;
	class NetParameter;
};

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
		eWaifu2xError_FailedCudaCheck,
	};

	enum eWaifu2xCudaError
	{
		eWaifu2xCudaError_OK = 0,
		eWaifu2xCudaError_NotFind,
		eWaifu2xCudaError_OldVersion,
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
	int input_block_size;
	// ブロック変換後の出力サイズ
	int output_size;
	// ネットワークに入力する画像のサイズ(出力画像の幅はlayer_num * 2だけ小さくなる)
	int block_width_height;
	// srcnn.prototxtで定義された入力する画像のサイズ
	int original_width_height;

	std::string mode;
	int noise_level;
	std::string model_dir;
	std::string process;

	int inner_padding;
	int outer_padding;

	int output_block_size;

	int input_plane;

	bool isCuda;

	boost::shared_ptr<caffe::Net<float>> net_noise;
	boost::shared_ptr<caffe::Net<float>> net_scale;

	float *input_block;
	float *dummy_data;
	float *output_block;

private:
	eWaifu2xError LoadMat(cv::Mat &float_image, const uint32_t* source, int width, int height);
	eWaifu2xError PaddingImage(const cv::Mat &input, cv::Mat &output);
	eWaifu2xError Zoom2xAndPaddingImage(const cv::Mat &input, cv::Mat &output, cv::Size_<int> &zoom_size);
	eWaifu2xError CreateZoomColorImage(const cv::Mat &float_image, const cv::Size_<int> &zoom_size, std::vector<cv::Mat> &cubic_planes);
	eWaifu2xError ConstractNet(boost::shared_ptr<caffe::Net<float>> &net, const std::string &model_path, const std::string &param_path, const std::string &process);
	eWaifu2xError SetParameter(caffe::NetParameter &param) const;
	eWaifu2xError ReconstructImage(boost::shared_ptr<caffe::Net<float>> net, cv::Mat &im);

public:
	Waifu2x();
	~Waifu2x();

	// mode: noise or scale or noise_scale or auto_scale
	// process: cpu or gpu or cudnn
	eWaifu2xError init(int argc, char** argv, const std::string &mode, const int noise_level, const std::string &model_dir, const std::string &process,
		const int crop_size = 128, const int batch_size = 1);

	void destroy();

	eWaifu2xError waifu2x(int factor, const uint32_t* source, uint32_t* dest, int width, int height);

	const std::string& used_process() const;
};
