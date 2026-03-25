#pragma once

#include <string>
#include "waifu2x.h"


class cNet
{
private:
	Waifu2x::eWaifu2xModelType mMode;

	std::shared_ptr<caffe::Net<float>> mNet;

	int mModelScale; // モデルが対象とする拡大率
	int mInnerScale; // ネット内部で拡大される倍率
	int mNetOffset; // ネットに入力するとどれくらい削れるか
	int mInputPlane; // ネットへの入力チャンネル数
	bool mHasNoiseScaleModel;

private:
	void LoadParamFromInfo(const Waifu2x::eWaifu2xModelType mode, const Waifu2x::stInfo& info);
	Waifu2x::eWaifu2xError LoadParameterFromJson(const std::filesystem::path& model_path, const std::filesystem::path& param_path
		, const std::filesystem::path& modelbin_path, const std::filesystem::path& caffemodel_path, const std::string& process);
	Waifu2x::eWaifu2xError SetParameter(caffe::NetParameter& param, const std::string& process) const;

public:
	cNet();
	~cNet();

	static Waifu2x::eWaifu2xError GetInfo(const std::filesystem::path& info_path, Waifu2x::stInfo& info);

	Waifu2x::eWaifu2xError ConstractNet(const Waifu2x::eWaifu2xModelType mode, const std::filesystem::path& model_path, const std::filesystem::path& param_path, const Waifu2x::stInfo& info, const std::string& process);

	int GetInputPlane() const;
	int GetInnerScale() const;
	int GetNetOffset() const;
	int GetScale() const;

	int GetInputMemorySize(const int crop_w, const int crop_h, const int outer_padding, const int batch_size) const;
	int GetOutputMemorySize(const int crop_w, const int crop_h, const int outer_padding, const int batch_size) const;

	Waifu2x::eWaifu2xError ReconstructImage(const bool UseTTA, const int crop_w, const int crop_h, const int outer_padding, const int batch_size, float* outputBlockBuf, const cv::Mat& inMat, cv::Mat& outMat);

	static std::string GetModelName(const std::filesystem::path& info_path);
};
