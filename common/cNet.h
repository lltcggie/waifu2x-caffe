#pragma once

#include <string>
#include "waifu2x.h"


struct stInfo
{
	struct stParam
	{
		int scale_factor;
		int offset;
	};

	std::string name;
	std::string arch_name;
	bool has_noise_scale;
	int channels;

	stParam noise;
	stParam scale;
	stParam noise_scale;
};

class cNet
{
private:
	Waifu2x::eWaifu2xModelType mMode;

	boost::shared_ptr<caffe::Net<float>> mNet;

	int mModelScale; // モデルが対象とする拡大率
	int mInnerScale; // ネット内部で拡大される倍率
	int mNetOffset; // ネットに入力するとどれくらい削れるか
	int mInputPlane; // ネットへの入力チャンネル数
	bool mHasNoiseScaleModel;

private:
	void LoadParamFromInfo(const Waifu2x::eWaifu2xModelType mode, const stInfo &info);
	Waifu2x::eWaifu2xError LoadParameterFromJson(const boost::filesystem::path &model_path, const boost::filesystem::path &param_path
		, const boost::filesystem::path &modelbin_path, const boost::filesystem::path &caffemodel_path, const std::string &process);
	Waifu2x::eWaifu2xError SetParameter(caffe::NetParameter &param, const std::string &process) const;

public:
	cNet();
	~cNet();

	static Waifu2x::eWaifu2xError GetInfo(const boost::filesystem::path &info_path, stInfo &info);

	Waifu2x::eWaifu2xError ConstractNet(const Waifu2x::eWaifu2xModelType mode, const boost::filesystem::path &model_path, const boost::filesystem::path &param_path, const stInfo &info, const std::string &process);

	int GetInputPlane() const;
	int GetInnerScale() const;
	int GetNetOffset() const;
	int GetScale() const;

	int GetInputMemorySize(const int crop_w, const int crop_h, const int outer_padding, const int batch_size) const;
	int GetOutputMemorySize(const int crop_w, const int crop_h, const int outer_padding, const int batch_size) const;

	Waifu2x::eWaifu2xError ReconstructImage(const bool UseTTA, const int crop_w, const int crop_h, const int outer_padding, const int batch_size, float *outputBlockBuf, const cv::Mat &inMat, cv::Mat &outMat);

	static std::string GetModelName(const boost::filesystem::path &info_path);
};
