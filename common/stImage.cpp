#include "stImage.h"
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

const int YToRGBConvertMode = CV_GRAY2RGB;
const int YToRGBConverInversetMode = CV_RGB2GRAY;
const int BGRToYConvertMode = CV_BGR2YUV;
const int BGRToConvertInverseMode = CV_YUV2BGR;

// floatな画像をuint8_tな画像に変換する際の四捨五入に使う値
// https://github.com/nagadomi/waifu2x/commit/797b45ae23665a1c5e3c481c018e48e6f0d0e383
const double clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5);
const double clip_eps16 = (1.0 / 65535.0) * 0.5 - (1.0e-7 * (1.0 / 65535.0) * 0.5);
const double clip_eps32 = 1.0 * 0.5 - (1.0e-7 * 0.5);

const std::vector<stImage::stOutputExtentionElement> stImage::OutputExtentionList =
{
	{L".png",{8, 16}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".bmp",{8}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".jpg",{8}, 0, 100, 95, cv::IMWRITE_JPEG_QUALITY},
	{L".jp2",{8, 16}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".sr",{8}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".tif",{8, 16, 32}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".hdr",{8, 16, 32}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".exr",{8, 16, 32}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".ppm",{8, 16}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".webp",{8}, 1, 100, 100, cv::IMWRITE_WEBP_QUALITY},
	{L".tga",{8}, 0, 1, 0, 0},
};


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
	boost::iostreams::stream<boost::iostreams::file_descriptor_source> is;

	try
	{
		is.open(path, std::ios_base::in | std::ios_base::binary);
	}
	catch (...)
	{
		return false;
	}

	return readFile(is, buf);
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
	boost::iostreams::stream<boost::iostreams::file_descriptor> os;

	try
	{
		os.open(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
	}
	catch (...)
	{
		return false;
	}

	return writeFile(os, buf);
}

static void Waifu2x_stbi_write_func(void *context, void *data, int size)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor> *osp = (boost::iostreams::stream<boost::iostreams::file_descriptor> *)context;
	osp->write((const char *)data, size);
}

int stImage::DepthBitToCVDepth(const int depth_bit)
{
	switch (depth_bit)
	{
	case 8:
		return CV_8U;

	case 16:
		return CV_16U;

	case 32:
		return CV_32F;
	}

	// 不明だけどとりあえずCV_8Uを返しておく
	return CV_8U;
}

double stImage::GetValumeMaxFromCVDepth(const int cv_depth)
{
	switch (cv_depth)
	{
	case CV_8U:
		return 255.0;

	case CV_16U:
		return 65535.0;

	case CV_32F:
		return 1.0;
	}

	// 不明だけどとりあえず255.0を返しておく
	return 255.0;
}

double stImage::GetEPS(const int cv_depth)
{
	switch (cv_depth)
	{
	case CV_8U:
		return clip_eps8;

	case CV_16U:
		return clip_eps16;

	case CV_32F:
		return clip_eps32;
	}

	// 不明だけどとりあえずclip_eps8返しておく
	return clip_eps8;
}


Waifu2x::eWaifu2xError stImage::AlphaMakeBorder(std::vector<cv::Mat> &planes, const cv::Mat &alpha, const int offset)
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

	return Waifu2x::eWaifu2xError_OK;
}

// 画像を読み込んで値を0.0f～1.0fの範囲に変換
Waifu2x::eWaifu2xError stImage::LoadMat(cv::Mat &im, const boost::filesystem::path &input_file)
{
	cv::Mat original_image;

	{
		std::vector<char> img_data;
		if (!readFile(input_file, img_data))
			return Waifu2x::eWaifu2xError_FailedOpenInputFile;

		const boost::filesystem::path ipext(input_file.extension());
		if (!boost::iequals(ipext.string(), ".bmp")) // 特定のファイル形式の場合OpenCVで読むとバグることがあるのでSTBIを優先させる
		{
			cv::Mat im(img_data.size(), 1, CV_8U, img_data.data());
			original_image = cv::imdecode(im, cv::IMREAD_UNCHANGED);

			if (original_image.empty())
			{
				const Waifu2x::eWaifu2xError ret = LoadMatBySTBI(original_image, img_data);
				if (ret != Waifu2x::eWaifu2xError_OK)
					return ret;
			}
		}
		else
		{
			const Waifu2x::eWaifu2xError ret = LoadMatBySTBI(original_image, img_data);
			if (ret != Waifu2x::eWaifu2xError_OK)
			{
				cv::Mat im(img_data.size(), 1, CV_8U, img_data.data());
				original_image = cv::imdecode(im, cv::IMREAD_UNCHANGED);
				if (original_image.empty())
					return ret;
			}
		}
	}

	im = original_image;

	return Waifu2x::eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError stImage::LoadMatBySTBI(cv::Mat &im, const std::vector<char> &img_data)
{
	int x, y, comp;
	stbi_uc *data = stbi_load_from_memory((const stbi_uc *)img_data.data(), img_data.size(), &x, &y, &comp, 0);
	if (!data)
		return Waifu2x::eWaifu2xError_FailedOpenInputFile;

	int type = 0;
	switch (comp)
	{
	case 1:
	case 3:
	case 4:
		type = CV_MAKETYPE(CV_8U, comp);
		break;

	default:
		return Waifu2x::eWaifu2xError_FailedOpenInputFile;
	}

	im = cv::Mat(cv::Size(x, y), type);

	const auto LinePixel = im.step1() / im.channels();
	const auto Channel = im.channels();
	const auto Width = im.size().width;
	const auto Height = im.size().height;

	assert(x == Width);
	assert(y == Height);
	assert(Channel == comp);

	auto ptr = im.data;
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

	return Waifu2x::eWaifu2xError_OK;
}

cv::Mat stImage::ConvertToFloat(const cv::Mat &im)
{
	cv::Mat convert;
	switch (im.depth())
	{
	case CV_8U:
		im.convertTo(convert, CV_32F, 1.0 / GetValumeMaxFromCVDepth(CV_8U));
		break;

	case CV_16U:
		im.convertTo(convert, CV_32F, 1.0 / GetValumeMaxFromCVDepth(CV_16U));
		break;

	case CV_32F:
		convert = im; // 元から0.0～1.0のはずなので変換は必要ない
		break;
	}

	return convert;
}


stImage::stImage() : mIsRequestDenoise(false), pad_w1(0), pad_h1(0), pad_w2(0), pad_h2(0)
{
}

stImage::~stImage()
{
}

void stImage::Clear()
{
	mOrgFloatImage.release();
	mTmpImageRGB.release();
	mTmpImageA.release();
	mEndImage.release();
}

Waifu2x::eWaifu2xError stImage::Load(const boost::filesystem::path &input_file)
{
	Clear();

	Waifu2x::eWaifu2xError ret;

	cv::Mat im;
	ret = LoadMat(im, input_file);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	mOrgFloatImage = im;
	mOrgChannel = im.channels();
	mOrgSize = im.size();

	const boost::filesystem::path ip(input_file);
	const boost::filesystem::path ipext(ip.extension());

	const bool isJpeg = boost::iequals(ipext.string(), ".jpg") || boost::iequals(ipext.string(), ".jpeg");

	mIsRequestDenoise = isJpeg;

	return Waifu2x::eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError stImage::Load(const void* source, const int width, const int height, const int channel, const int stride)
{
	Clear();

	cv::Mat original_image(cv::Size(width, height), CV_MAKETYPE(CV_8U, channel), (void *)source, stride);

	if (original_image.channels() >= 3) // RGBなのでBGRにする
	{
		std::vector<cv::Mat> planes;
		cv::split(original_image, planes);

		std::swap(planes[0], planes[2]);

		cv::merge(planes, original_image);
	}

	mOrgFloatImage = original_image;
	mOrgChannel = original_image.channels();
	mOrgSize = original_image.size();

	mIsRequestDenoise = false;

	return Waifu2x::eWaifu2xError_OK;
}

double stImage::GetScaleFromWidth(const int width) const
{
	return (double)width / (double)mOrgSize.width;
}

double stImage::GetScaleFromHeight(const int height) const
{
	return (double)height / (double)mOrgSize.height;
}

bool stImage::RequestDenoise() const
{
	return mIsRequestDenoise;
}

void stImage::Preprocess(const int input_plane, const int net_offset)
{
	mOrgFloatImage = ConvertToFloat(mOrgFloatImage);

	ConvertToNetFormat(input_plane, net_offset);
}

void stImage::ConvertToNetFormat(const int input_plane, const int alpha_offset)
{
	if (input_plane == 1) // Yモデル
	{
		if (mOrgFloatImage.channels() == 1) // 1chだけなのでそのまま
			mTmpImageRGB = mOrgFloatImage;
		else // BGRなので変換
		{
			mTmpImageRGB = mOrgFloatImage;

			if (mTmpImageRGB.channels() == 4) // BGRAなのでAだけ取り出す
			{
				std::vector<cv::Mat> planes;
				cv::split(mTmpImageRGB, planes);

				mTmpImageA = planes[3];
				planes.resize(3);

				AlphaMakeBorder(planes, mTmpImageA, alpha_offset); // 透明なピクセルと不透明なピクセルの境界部分の色を広げる

				// CreateBrightnessImage()でBGRからYに変換するので特にRGBに変えたりはしない
				cv::merge(planes, mTmpImageRGB);
			}

			CreateBrightnessImage(mTmpImageRGB, mTmpImageRGB);
		}
	}
	else // RGBモデル
	{
		if (mOrgFloatImage.channels() == 1) // 1chだけなのでRGBに変換
		{
			cv::cvtColor(mOrgFloatImage, mTmpImageRGB, YToRGBConvertMode);
			mOrgFloatImage.release();
		}
		else // BGRからRGBに変換(AがあったらAも取り出す)
		{
			std::vector<cv::Mat> planes;
			cv::split(mOrgFloatImage, planes);
			mOrgFloatImage.release();

			if (planes.size() == 4) // BGRAなのでAだけ取り出す
			{
				mTmpImageA = planes[3];
				planes.resize(3);

				AlphaMakeBorder(planes, mTmpImageA, alpha_offset); // 透明なピクセルと不透明なピクセルの境界部分の色を広げる

				// α拡大用にRGBに変換
				cv::cvtColor(mTmpImageA, mTmpImageA, CV_GRAY2RGB);
			}

			// BGRからRGBにする
			std::swap(planes[0], planes[2]);

			cv::merge(planes, mTmpImageRGB);
		}

		mOrgFloatImage.release();
	}
}

// 画像から輝度の画像を取り出す
Waifu2x::eWaifu2xError stImage::CreateBrightnessImage(const cv::Mat &float_image, cv::Mat &im)
{
	if (float_image.channels() > 1)
	{
		cv::Mat converted_color;
		cv::cvtColor(float_image, converted_color, BGRToYConvertMode);

		std::vector<cv::Mat> planes;
		cv::split(converted_color, planes);

		im = planes[0];
		planes.clear();
	}
	else
		im = float_image;

	return Waifu2x::eWaifu2xError_OK;
}

bool stImage::HasAlpha() const
{
	return !mTmpImageA.empty();
}

void stImage::GetScalePaddingedRGB(cv::Mat &im, cv::Size_<int> &size, const int net_offset, const int outer_padding,
	const int crop_w, const int crop_h, const int scale)
{
	GetScalePaddingedImage(mTmpImageRGB, im, size, net_offset, outer_padding, crop_w, crop_h, scale);
}

void stImage::SetReconstructedRGB(cv::Mat &im, const cv::Size_<int> &size, const int inner_scale)
{
	SetReconstructedImage(mTmpImageRGB, im, size, inner_scale);
}

void stImage::GetScalePaddingedA(cv::Mat &im, cv::Size_<int> &size, const int net_offset, const int outer_padding,
	const int crop_w, const int crop_h, const int scale)
{
	GetScalePaddingedImage(mTmpImageA, im, size, net_offset, outer_padding, crop_w, crop_h, scale);
}

void stImage::SetReconstructedA(cv::Mat &im, const cv::Size_<int> &size, const int inner_scale)
{
	SetReconstructedImage(mTmpImageA, im, size, inner_scale);
}

void stImage::GetScalePaddingedImage(cv::Mat &in, cv::Mat &out, cv::Size_<int> &size, const int net_offset, const int outer_padding,
	const int crop_w, const int crop_h, const int scale)
{
	cv::Mat ret;

	if (scale > 1)
	{
		cv::Size_<int> zoom_size = in.size();
		zoom_size.width *= scale;
		zoom_size.height *= scale;

		cv::resize(in, ret, zoom_size, 0.0, 0.0, cv::INTER_NEAREST);
	}
	else
		ret = in;

	in.release();

	size = ret.size();

	PaddingImage(ret, net_offset, outer_padding, crop_w, crop_h, ret);

	out = ret;
}

// 入力画像の(Photoshopでいう)キャンバスサイズをoutput_sizeの倍数に変更
// 画像は左上配置、余白はcv::BORDER_REPLICATEで埋める
void stImage::PaddingImage(const cv::Mat &input, const int net_offset, const int outer_padding,
	const int crop_w, const int crop_h, cv::Mat &output)
{
	const auto pad_w1 = net_offset + outer_padding;
	const auto pad_h1 = net_offset + outer_padding;
	const auto pad_w2 = (int)ceil((double)input.size().width / (double)crop_w) * crop_w - input.size().width + net_offset + outer_padding;
	const auto pad_h2 = (int)ceil((double)input.size().height / (double)crop_h) * crop_h - input.size().height + net_offset + outer_padding;

	cv::copyMakeBorder(input, output, pad_h1, pad_h2, pad_w1, pad_w2, cv::BORDER_REPLICATE);
}

// 拡大、パディングされた画像を設定
void stImage::SetReconstructedImage(cv::Mat &dst, cv::Mat &src, const cv::Size_<int> &size, const int inner_scale)
{
	const cv::Size_<int> s(size * inner_scale);

	// ブロックサイズ用のパディングを取り払う(outer_paddingは再構築の過程で取り除かれている)
	dst = src(cv::Rect(0, 0, s.width, s.height));

	src.release();
}

void stImage::Postprocess(const int input_plane, const double scale, const int depth)
{
	DeconvertFromNetFormat(input_plane);
	ShrinkImage(scale);

	// 値を0～1にクリッピング
	cv::threshold(mEndImage, mEndImage, 1.0, 1.0, cv::THRESH_TRUNC);
	cv::threshold(mEndImage, mEndImage, 0.0, 0.0, cv::THRESH_TOZERO);

	mEndImage = DeconvertFromFloat(mEndImage, depth);

	AlphaCleanImage(mEndImage);
}

void stImage::DeconvertFromNetFormat(const int input_plane)
{
	if (input_plane == 1) // Yモデル
	{
		if (mOrgChannel == 1) // もともと1chだけなのでそのまま
		{
			mEndImage = mTmpImageRGB;
			mTmpImageRGB.release();
			mOrgFloatImage.release();
		}
		else // もともとBGRなので既存アルゴリズムで拡大したUVに拡大したYを合体して戻す
		{
			std::vector<cv::Mat> color_planes;
			CreateZoomColorImage(mOrgFloatImage, mTmpImageRGB.size(), color_planes);
			mOrgFloatImage.release();

			color_planes[0] = mTmpImageRGB;
			mTmpImageRGB.release();

			cv::Mat converted_image;
			cv::merge(color_planes, converted_image);
			color_planes.clear();

			cv::cvtColor(converted_image, mEndImage, BGRToConvertInverseMode);
			converted_image.release();

			if (!mTmpImageA.empty()) // Aもあるので合体
			{
				std::vector<cv::Mat> planes;
				cv::split(mEndImage, planes);

				planes.push_back(mTmpImageA);
				mTmpImageA.release();

				cv::merge(planes, mEndImage);
			}
		}
	}
	else // RGBモデル
	{
		// ここの地点でmOrgFloatImageは空

		if (mOrgChannel == 1) // もともと1chだけなので戻す
		{
			cv::cvtColor(mTmpImageRGB, mEndImage, YToRGBConverInversetMode);
			mTmpImageRGB.release();
		}
		else // もともとBGRなのでRGBから戻す(AがあったらAも合体して戻す)
		{
			std::vector<cv::Mat> planes;
			cv::split(mTmpImageRGB, planes);
			mTmpImageRGB.release();

			if (!mTmpImageA.empty()) // Aもあるので合体
			{
				// RGBから1chに戻す
				cv::cvtColor(mTmpImageA, mTmpImageA, CV_RGB2GRAY);

				planes.push_back(mTmpImageA);
				mTmpImageA.release();
			}

			// RGBからBGRにする
			std::swap(planes[0], planes[2]);

			cv::merge(planes, mEndImage);
		}
	}
}

void stImage::ShrinkImage(const double scale)
{
	// TODO: scale = 1.0 でも悪影響を及ぼさないか調べる

	const int scaleBase = 2; // TODO: モデルの拡大率によって可変できるようにする

	const int scaleNum = ceil(log(scale) / log(scaleBase));
	const double shrinkRatio = scale >= 1.0 ? scale / std::pow(scaleBase, scaleNum) : scale;

	const cv::Size_<int> ns(mOrgSize.width * scale, mOrgSize.height * scale);
	if (mEndImage.size().width != ns.width || mEndImage.size().height != ns.height)
	{
		int argo = cv::INTER_CUBIC;
		if (scale < 0.5)
			argo = cv::INTER_AREA;

		cv::resize(mEndImage, mEndImage, ns, 0.0, 0.0, argo);
	}
}

cv::Mat stImage::DeconvertFromFloat(const cv::Mat &im, const int depth)
{
	const int cv_depth = DepthBitToCVDepth(depth);
	const double max_val = GetValumeMaxFromCVDepth(cv_depth);
	const double eps = GetEPS(cv_depth);

	cv::Mat ret;
	if (depth == 32) // 出力がfloat形式なら変換しない
		ret = im;
	else
		im.convertTo(ret, cv_depth, max_val, eps);

	return ret;
}

namespace
{
	template<typename T>
	void AlphaZeroToZero(std::vector<cv::Mat> &planes)
	{
		cv::Mat alpha(planes[3]);

		const T *aptr = (const T *)alpha.data;

		T *ptr0 = (T *)planes[0].data;
		T *ptr1 = (T *)planes[1].data;
		T *ptr2 = (T *)planes[2].data;

		const size_t Line = alpha.step1();
		const size_t Width = alpha.size().width;
		const size_t Height = alpha.size().height;

		for (size_t i = 0; i < Height; i++)
		{
			for (size_t j = 0; j < Width; j++)
			{
				const size_t pos = Line * i + j;

				if (aptr[pos] == (T)0)
					ptr0[pos] = ptr1[pos] = ptr2[pos] = (T)0;
			}
		}
	}
}

void stImage::AlphaCleanImage(cv::Mat &im)
{
	// 完全透明のピクセルの色を消す(処理の都合上、完全透明のピクセルにも色を付けたから)
	// モデルによっては画像全域の完全透明の場所にごく小さい値のアルファが広がることがある。それを消すためにcv_depthへ変換してからこの処理を行うことにした
	// (ただしcv_depthが32の場合だと意味は無いが)
	// TODO: モデル(例えばPhoto)によっては0しかない画像を変換しても0.000114856390とかになるので、適切な値のクリッピングを行う？
	if (im.channels() > 3)
	{
		std::vector<cv::Mat> planes;
		cv::split(im, planes);
		im.release();

		const auto depth = planes[0].depth();
		switch (depth)
		{
		case CV_8U:
			AlphaZeroToZero<uint8_t>(planes);
			break;

		case CV_16U:
			AlphaZeroToZero<uint16_t>(planes);
			break;

		case CV_32F:
			AlphaZeroToZero<float>(planes);
			break;

		case CV_64F:
			AlphaZeroToZero<double>(planes);
			break;
		}

		cv::merge(planes, im);
	}
}


// 入力画像をzoom_sizeの大きさにcv::INTER_CUBICで拡大し、色情報のみを残す
Waifu2x::eWaifu2xError stImage::CreateZoomColorImage(const cv::Mat &float_image, const cv::Size_<int> &zoom_size, std::vector<cv::Mat> &cubic_planes)
{
	cv::Mat zoom_cubic_image;
	cv::resize(float_image, zoom_cubic_image, zoom_size, 0.0, 0.0, cv::INTER_CUBIC);

	cv::Mat converted_cubic_image;
	cv::cvtColor(zoom_cubic_image, converted_cubic_image, BGRToYConvertMode);
	zoom_cubic_image.release();

	cv::split(converted_cubic_image, cubic_planes);
	converted_cubic_image.release();

	// このY成分は使わないので解放
	cubic_planes[0].release();

	return Waifu2x::eWaifu2xError_OK;
}

cv::Mat stImage::GetEndImage() const
{
	return mEndImage;
}

Waifu2x::eWaifu2xError stImage::Save(const boost::filesystem::path &output_file, const boost::optional<int> &output_quality)
{
	return WriteMat(mEndImage, output_file, output_quality);
}

Waifu2x::eWaifu2xError stImage::WriteMat(const cv::Mat &im, const boost::filesystem::path &output_file, const boost::optional<int> &output_quality)
{
	const boost::filesystem::path ip(output_file);
	const std::string ext = ip.extension().string();

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

		boost::iostreams::stream<boost::iostreams::file_descriptor> os;

		try
		{
			os.open(output_file, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
		}
		catch (...)
		{
			return Waifu2x::eWaifu2xError_FailedOpenOutputFile;
		}

		if (!os)
			return Waifu2x::eWaifu2xError_FailedOpenOutputFile;

		// RLE圧縮の設定
		bool isSet = false;
		const auto &OutputExtentionList = stImage::OutputExtentionList;
		for (const auto &elm : OutputExtentionList)
		{
			if (elm.ext == L".tga")
			{
				if (elm.imageQualitySettingVolume && output_quality)
				{
					stbi_write_tga_with_rle = *output_quality;
					isSet = true;
				}

				break;
			}
		}

		// 設定されなかったのでデフォルトにする
		if (!isSet)
			stbi_write_tga_with_rle = 1;

		if (!stbi_write_tga_to_func(Waifu2x_stbi_write_func, &os, im.size().width, im.size().height, im.channels(), data))
			return Waifu2x::eWaifu2xError_FailedOpenOutputFile;

		return Waifu2x::eWaifu2xError_OK;
	}

	try
	{
		const boost::filesystem::path op(output_file);
		const boost::filesystem::path opext(op.extension());

		std::vector<int> params;

		const auto &OutputExtentionList = stImage::OutputExtentionList;
		for (const auto &elm : OutputExtentionList)
		{
			if (elm.ext == opext)
			{
				if (elm.imageQualitySettingVolume && output_quality)
				{
					params.push_back(*elm.imageQualitySettingVolume);
					params.push_back(*output_quality);
				}

				break;
			}
		}

		std::vector<uchar> buf;
		cv::imencode(ext, im, buf, params);

		if (writeFile(output_file, buf))
			return Waifu2x::eWaifu2xError_OK;

	}
	catch (...)
	{
	}

	return Waifu2x::eWaifu2xError_FailedOpenOutputFile;
}
