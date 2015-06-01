#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <utility>
#include <functional>

enum eWaifu2xError
{
	eWaifu2xError_OK = 0,
	eWaifu2xError_Cancel,
	eWaifu2xError_InvalidParameter,
	eWaifu2xError_FailedOpenInputFile,
	eWaifu2xError_FailedOpenOutputFile,
	eWaifu2xError_FailedOpenModelFile,
	eWaifu2xError_FailedParseModelFile,
	eWaifu2xError_FailedConstructModel,
	eWaifu2xError_FailedProcessCaffe,
};

typedef std::pair<std::string, std::string> InputOutputPathPair;
typedef std::pair<InputOutputPathPair, eWaifu2xError> PathAndErrorPair;
typedef std::function<bool()> waifu2xCancelFunc;
typedef std::function<void(const int ProgressFileMax, const int ProgressFileNow)> waifu2xProgressFunc;
typedef std::function<void(const uint64_t InitTime, const uint64_t cuDNNCheckTime, const uint64_t ProcessTime)> waifu2xTimeFunc;

bool can_use_cuDNN();

// mode: noise or scale or noise_scale or auto_scale
// process: cpu or gpu or cudnn
eWaifu2xError waifu2x(int argc, char** argv,
	const std::vector<InputOutputPathPair> &file_paths, const std::string &mode, const int noise_level, const double scale_ratio, const std::string &model_dir, const std::string &process,
	std::vector<PathAndErrorPair> &errors, const waifu2xCancelFunc cancel_func = nullptr, const waifu2xProgressFunc progress_func = nullptr, const waifu2xTimeFunc time_func = nullptr);
