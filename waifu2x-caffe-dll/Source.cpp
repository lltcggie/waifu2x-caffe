#include <stdio.h>
#include <stdlib.h>
#include "../common/waifu2x.h"


__declspec(dllexport)
void* Waifu2xInit(const char *mode, const int noise_level, const char *model_dir, const char *process, const int output_depth = 8, const bool use_tta = false, const int crop_size = 128, const int batch_size = 1)
{
	Waifu2x *obj = new Waifu2x();

	Waifu2x::eWaifu2xModelType mt;
	if (strcmp("noise", mode) == 0)
		mt = Waifu2x::eWaifu2xModelTypeNoise;
	else if (strcmp("scale", mode) == 0)
		mt = Waifu2x::eWaifu2xModelTypeScale;
	else if (strcmp("noise_scale", mode) == 0)
		mt = Waifu2x::eWaifu2xModelTypeNoiseScale;
	else if (strcmp("auto_scale", mode) == 0)
		mt = Waifu2x::eWaifu2xModelTypeAutoScale;

	// if (obj->Init(1, argv, mode, noise_level, 2.0, boost::optional<int>(), boost::optional<int>(), model_dir, process, boost::optional<int>(), output_depth, use_tta, crop_size, batch_size) != Waifu2x::eWaifu2xError_OK)
	if (obj->Init(mt, noise_level, model_dir, process) != Waifu2x::eWaifu2xError_OK)
	{
		delete obj;
		return nullptr;
	}

	return obj;
}

__declspec(dllexport)
bool Waifu2xProcess(void *waifu2xObj, double factor, const void* source, void* dest, int width, int height, int in_channel, int in_stride, int out_channel, int out_stride)
{
	if (!waifu2xObj)
		return false;

	Waifu2x *obj = (Waifu2x *)waifu2xObj;

	return obj->waifu2x(factor, source, dest, width, height, in_channel, in_stride, out_channel, out_stride) == Waifu2x::eWaifu2xError_OK;
}

__declspec(dllexport)
void Waifu2xDestory(void *waifu2xObj)
{
	if (waifu2xObj)
	{
		Waifu2x *obj = (Waifu2x *)waifu2xObj;
		delete obj;
	}
}
