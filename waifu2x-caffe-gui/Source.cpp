#define _CRT_SECURE_NO_WARNINGS
#include <glog/logging.h>
#include <windows.h>
#include <Commctrl.h>
#include <tchar.h>
#include <stdio.h>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <boost/math/common_factor_rt.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include <cblas.h>
#include <dlgs.h>
#include "resource.h"
#include "tstring.h"
#include "LangStringList.h"
#include "../common/waifu2x.h"

#include "CDialog.h"
#include "CControl.h"

#pragma comment(lib, "winmm.lib")

#undef ERROR

#define WM_FAILD_CREATE_DIR (WM_APP + 5)
#define WM_ON_WAIFU2X_ERROR (WM_APP + 6)
#define WM_END_THREAD (WM_APP + 7)

const size_t AR_PATH_MAX(1024);

const int MinCommonDivisor = 50;
const int DefaultCommonDivisor = 128;
const std::pair<int, int> DefaultCommonDivisorRange = {90, 140};

const TCHAR * const CropSizeListName = TEXT("crop_size_list.txt");
const TCHAR * const SettingFileName = TEXT("setting.ini");
const TCHAR * const LangDir = TEXT("lang");
const TCHAR * const LangListFileName = TEXT("lang/LangList.txt");

const TCHAR * const MultiFileStr = TEXT("(Multi File)");

const UINT_PTR nIDEventTimeLeft = 1000;


// http://stackoverflow.com/questions/10167382/boostfilesystem-get-relative-path
boost::filesystem::path relativePath(const boost::filesystem::path &path, const boost::filesystem::path &relative_to)
{
	// create absolute paths
	boost::filesystem::path p = boost::filesystem::absolute(path);
	boost::filesystem::path r = boost::filesystem::absolute(relative_to);

	// if root paths are different, return absolute path
	if (p.root_path() != r.root_path())
		return p;

	// initialize relative path
	boost::filesystem::path result;

	// find out where the two paths diverge
	boost::filesystem::path::const_iterator itr_path = p.begin();
	boost::filesystem::path::const_iterator itr_relative_to = r.begin();
	while (*itr_path == *itr_relative_to && itr_path != p.end() && itr_relative_to != r.end()) {
		++itr_path;
		++itr_relative_to;
	}

	// add "../" for each remaining token in relative_to
	if (itr_relative_to != r.end()) {
		++itr_relative_to;
		while (itr_relative_to != r.end()) {
			result /= "..";
			++itr_relative_to;
		}
	}

	// add remaining path
	while (itr_path != p.end()) {
		result /= *itr_path;
		++itr_path;
	}

	return result;
}

std::vector<int> CommonDivisorList(const int N)
{
	std::vector<int> list;

	const int sq = sqrt(N);
	for (int i = 1; i <= sq; i++)
	{
		if (N % i == 0)
			list.push_back(i);
	}

	const int sqs = list.size();
	for (int i = 0; i < sqs; i++)
		list.push_back(N / list[i]);

	std::sort(list.begin(), list.end());

	return list;
}

// ダイアログ用
class DialogEvent
{
private:
	HWND dh;

	boost::filesystem::path exeDir;
	std::vector<int> CropSizeList;

	tstring input_str;
	std::vector<tstring> input_str_multi;
	tstring output_str;
	std::string mode;
	int noise_level;
	double scale_ratio;
	int scale_width;
	int scale_height;
	tstring model_dir;
	std::string process;
	tstring outputExt;
	tstring inputFileExt;

	bool use_tta;

	boost::optional<int> output_quality;
	int output_depth;

	int crop_size;
	int batch_size;

	std::vector<tstring> extList;

	std::thread processThread;
	std::atomic_bool cancelFlag;

	tstring autoSetAddName;
	bool isLastError;

	tstring logMessage;

	std::string usedProcess;
	std::chrono::system_clock::duration cuDNNCheckTime;
	std::chrono::system_clock::duration InitTime;
	std::chrono::system_clock::duration ProcessTime;

	enum eScaleType
	{
		eScaleTypeRatio,
		eScaleTypeWidth,
		eScaleTypeHeight,
		eScaleTypeEnd,
	};

	eScaleType scaleType;

	enum eModelType
	{
		eModelTypeRGB,
		eModelTypePhoto,
		eModelTypeY,
		eModelTypeEnd,
	};

	eModelType modelType;

	LangStringList langStringList;
	std::wstring LangName;

	std::atomic<int64_t> TimeLeftThread;
	std::atomic<DWORD> TimeLeftGetTimeThread;

	bool isCommandLineStart;

private:
	template<typename T>
	static tstring to_tstring(T val)
	{
#ifdef UNICODE
		return std::to_wstring(val);
#else
		return std::to_string(val);
#endif
	}

	tstring AddName() const
	{
		tstring addstr;

		addstr += TEXT("(");
		switch (modelType)
		{
		case eModelTypeRGB:
			addstr += TEXT("RGB");
			break;

		case eModelTypePhoto:
			addstr += TEXT("Photo");
			break;

		case eModelTypeY:
			addstr += TEXT("Y");
			break;
		}
		addstr += TEXT(")");

		addstr += TEXT("(");
		if (mode == "noise")
			addstr += TEXT("noise");
		else if (mode == "scale")
			addstr += TEXT("scale");
		else if (mode == "noise_scale")
			addstr += TEXT("noise_scale");
		else if (mode == "auto_scale")
			addstr += TEXT("auto_scale");
		addstr += TEXT(")");

		if (mode.find("noise") != mode.npos || mode.find("auto_scale") != mode.npos)
			addstr += TEXT("(Level") + to_tstring(noise_level) + TEXT(")");
		if (use_tta)
			addstr += TEXT("(tta)");

		if (mode.find("scale") != mode.npos)
		{
			if (scaleType == eScaleTypeRatio)
				addstr += TEXT("(x") + to_tstring(scale_ratio) + TEXT(")");
			else if (scaleType == eScaleTypeWidth)
				addstr += TEXT("(width ") + to_tstring(scale_width) + TEXT(")");
			else
				addstr += TEXT("(height ") + to_tstring(scale_height) + TEXT(")");
		}

		if (output_depth != 8)
			addstr += TEXT("(") + boost::lexical_cast<tstring>(output_depth) + TEXT("bit)");

		return addstr;
	}

	bool SyncMember(const bool NotSyncCropSize, const bool silent = false)
	{
		bool ret = true;

		if(input_str_multi.size() == 0)
		{
			TCHAR buf[AR_PATH_MAX] = TEXT("");
			GetWindowText(GetDlgItem(dh, IDC_EDIT_INPUT), buf, _countof(buf));
			buf[_countof(buf) - 1] = TEXT('\0');

			input_str = buf;
		}

		{
			TCHAR buf[AR_PATH_MAX] = TEXT("");
			GetWindowText(GetDlgItem(dh, IDC_EDIT_OUTPUT), buf, _countof(buf));
			buf[_countof(buf) - 1] = TEXT('\0');

			output_str = buf;
		}

		if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODE_NOISE), BM_GETCHECK, 0, 0))
			mode = "noise";
		else if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODE_SCALE), BM_GETCHECK, 0, 0))
			mode = "scale";
		else if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODE_NOISE_SCALE), BM_GETCHECK, 0, 0))
			mode = "noise_scale";
		else
			mode = "auto_scale";

		if (SendMessage(GetDlgItem(dh, IDC_RADIONOISE_LEVEL1), BM_GETCHECK, 0, 0))
			noise_level = 1;
		else
			noise_level = 2;

		if (SendMessage(GetDlgItem(dh, IDC_RADIO_SCALE_RATIO), BM_GETCHECK, 0, 0))
			scaleType = eScaleTypeRatio;
		else if(SendMessage(GetDlgItem(dh, IDC_RADIO_SCALE_WIDTH), BM_GETCHECK, 0, 0))
			scaleType = eScaleTypeWidth;
		else
			scaleType = eScaleTypeHeight;

		{
			TCHAR buf[AR_PATH_MAX] = TEXT("");
			GetWindowText(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), buf, _countof(buf));
			buf[_countof(buf) - 1] = TEXT('\0');

			TCHAR *ptr = nullptr;
			const double d = _tcstod(buf, &ptr);
			if (!ptr || *ptr != TEXT('\0') || d <= 0.0)
			{
				if (scaleType == eScaleTypeRatio)
				{
					ret = false;

					if(!silent)
						MessageBox(dh, langStringList.GetString(L"MessageScaleRateCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
				}
			}
			else
				scale_ratio = d;
		}

		{
			TCHAR buf[AR_PATH_MAX] = TEXT("");
			GetWindowText(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), buf, _countof(buf));
			buf[_countof(buf) - 1] = TEXT('\0');

			TCHAR *ptr = nullptr;
			const long l = _tcstol(buf, &ptr, 10);
			if (!ptr || *ptr != TEXT('\0') || l <= 0)
			{
				if (scaleType == eScaleTypeWidth)
				{
					ret = false;

					if (!silent)
						MessageBox(dh, langStringList.GetString(L"MessageScaleWidthCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
				}
			}
			else
				scale_width = l;
		}

		{
			TCHAR buf[AR_PATH_MAX] = TEXT("");
			GetWindowText(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), buf, _countof(buf));
			buf[_countof(buf) - 1] = TEXT('\0');

			TCHAR *ptr = nullptr;
			const long l = _tcstol(buf, &ptr, 10);
			if (!ptr || *ptr != TEXT('\0') || l <= 0)
			{
				if (scaleType == eScaleTypeHeight)
				{
					ret = false;

					if (!silent)
						MessageBox(dh, langStringList.GetString(L"MessageScaleHeightCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
				}
			}
			else
				scale_height = l;
		}

		if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODEL_RGB), BM_GETCHECK, 0, 0))
		{
			model_dir = TEXT("models/anime_style_art_rgb");
			modelType = eModelTypeRGB;
		}
		else if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODEL_Y), BM_GETCHECK, 0, 0))
		{
			model_dir = TEXT("models/anime_style_art");
			modelType = eModelTypeY;
		}
		else
		{
			model_dir = TEXT("models/photo");
			modelType = eModelTypePhoto;
		}

		{
			const auto &OutputExtentionList = Waifu2x::OutputExtentionList;

			const int cur = SendMessage(GetDlgItem(dh, IDC_COMBO_OUT_EXT), CB_GETCURSEL, 0, 0);
			if (cur < 0 || cur >= OutputExtentionList.size())
				MessageBox(dh, langStringList.GetString(L"MessageOutputExtentionCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
			else
			{
				const auto elm = OutputExtentionList[cur];

				outputExt = elm.ext;

				TCHAR buf[AR_PATH_MAX] = TEXT("");

				GetWindowText(GetDlgItem(dh, IDC_EDIT_OUT_QUALITY), buf, _countof(buf));
				buf[_countof(buf) - 1] = TEXT('\0');

				if (elm.imageQualityStart && elm.imageQualityEnd)
				{
					TCHAR *ptr = nullptr;
					const auto num = _tcstol(buf, &ptr, 10);
					if (!ptr || *ptr != '\0' || num < *elm.imageQualityStart || num > *elm.imageQualityEnd)
					{
						output_quality.reset();
						ret = false;

						MessageBox(dh, langStringList.GetString(L"MessageOutputQualityCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
					}
					else
						output_quality = num;
				}

				const int curDepth = SendMessage(GetDlgItem(dh, IDC_COMBO_OUTPUT_DEPTH), CB_GETCURSEL, 0, 0);
				if (curDepth < 0 || curDepth >= elm.depthList.size())
					MessageBox(dh, langStringList.GetString(L"MessageOutputQualityCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
				else
					output_depth = elm.depthList[curDepth];
			}
		}

		if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODE_CPU), BM_GETCHECK, 0, 0))
			process = "cpu";
		else
			process = "gpu";

		{
			TCHAR buf[AR_PATH_MAX] = TEXT("");
			GetWindowText(GetDlgItem(dh, IDC_EDIT_INPUT_EXT_LIST), buf, _countof(buf));
			buf[_countof(buf) - 1] = TEXT('\0');

			inputFileExt = buf;

			// input_extention_listを文字列の配列にする

			typedef boost::char_separator<TCHAR> char_separator;
			typedef boost::tokenizer<char_separator, tstring::const_iterator, tstring> tokenizer;

			char_separator sep(TEXT(":"), TEXT(""), boost::drop_empty_tokens);
			tokenizer tokens(inputFileExt, sep);

			for (auto tok_iter = tokens.begin(); tok_iter != tokens.end(); ++tok_iter)
			{
				tstring ext(*tok_iter);
				std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
				extList.push_back(TEXT(".") + ext);
			}
		}

		if (!NotSyncCropSize)
		{
			TCHAR buf[AR_PATH_MAX] = TEXT("");
			GetWindowText(GetDlgItem(dh, IDC_COMBO_CROP_SIZE), buf, _countof(buf));
			buf[_countof(buf) - 1] = TEXT('\0');

			TCHAR *ptr = nullptr;
			crop_size = _tcstol(buf, &ptr, 10);
			if (!ptr || *ptr != '\0' || crop_size <= 0)
			{
				crop_size = 128;
				ret = false;

				MessageBox(dh, langStringList.GetString(L"MessageCropSizeCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
			}
		}

		use_tta = SendMessage(GetDlgItem(dh, IDC_CHECK_TTA), BM_GETCHECK, 0, 0) == BST_CHECKED;

		return ret;
	}

	void SetCropSizeList(const boost::filesystem::path &input_path)
	{
		HWND hcrop = GetDlgItem(dh, IDC_COMBO_CROP_SIZE);

		int gcd = 1;
		if (boost::filesystem::exists(input_path) && !boost::filesystem::is_directory(input_path))
		{
			auto mat = Waifu2x::LoadMat(input_path.string());
			if (mat.empty())
				return;

			auto size = mat.size();
			mat.release();

			gcd = boost::math::gcd(size.width, size.height);
		}

		while (SendMessage(hcrop, CB_GETCOUNT, 0, 0) != 0)
			SendMessage(hcrop, CB_DELETESTRING, 0, 0);

		// 最大公約数の約数のリスト取得
		std::vector<int> list(CommonDivisorList(gcd));

		// MinCommonDivisor未満の約数削除
		list.erase(std::remove_if(list.begin(), list.end(), [](const int v)
		{
			return v < MinCommonDivisor;
		}
		), list.end());

		int mindiff = INT_MAX;
		int defaultIndex = -1;
		for (int i = 0; i < list.size(); i++)
		{
			const int n = list[i];

			tstring str(to_tstring(n));
			SendMessage(hcrop, CB_ADDSTRING, 0, (LPARAM)str.c_str());

			const int diff = abs(DefaultCommonDivisor - n);
			if (DefaultCommonDivisorRange.first <= n && n <= DefaultCommonDivisorRange.second && diff < mindiff)
			{
				mindiff = diff;
				defaultIndex = i;
			}
		}

		SendMessage(hcrop, CB_ADDSTRING, 0, (LPARAM)TEXT("-----------------------"));

		// CropSizeListの値を追加していく
		mindiff = INT_MAX;
		int defaultListIndex = -1;
		for (const auto n : CropSizeList)
		{
			tstring str(to_tstring(n));
			const int index = SendMessage(hcrop, CB_ADDSTRING, 0, (LPARAM)str.c_str());

			const int diff = abs(DefaultCommonDivisor - n);
			if (DefaultCommonDivisorRange.first <= n && n <= DefaultCommonDivisorRange.second && diff < mindiff)
			{
				mindiff = diff;
				defaultListIndex = index;
			}
		}

		if (defaultIndex == -1)
			defaultIndex = defaultListIndex;

		if (GetWindowTextLength(hcrop) == 0)
			SendMessage(hcrop, CB_SETCURSEL, defaultIndex, 0);
	}

	static boost::filesystem::path GetFileName(const boost::filesystem::path &input_path)
	{
		if (boost::filesystem::is_directory(input_path))
			return input_path.stem();
		else
			return input_path.filename();
	}

	void ProcessWaifu2x()
	{
		std::vector<std::pair<tstring, tstring>> file_paths;

		const auto inputFunc = [this, &file_paths](const tstring &input)
		{
			const boost::filesystem::path input_path(boost::filesystem::absolute(input));

			if (boost::filesystem::is_directory(input_path)) // input_pathがフォルダならそのディレクトリ以下の画像ファイルを一括変換
			{
				boost::filesystem::path output_path(output_str);

				output_path = boost::filesystem::absolute(output_path);

				if (!boost::filesystem::exists(output_path))
				{
					if (!boost::filesystem::create_directory(output_path))
					{
						SendMessage(dh, WM_FAILD_CREATE_DIR, (WPARAM)&output_path, 0);
						PostMessage(dh, WM_END_THREAD, 0, 0);
						// printf("出力フォルダ「%s」の作成に失敗しました\n", output_path.string().c_str());
						return;
					}
				}

				// 変換する画像の入力、出力パスを取得
				const auto func = [this, &input_path, &output_path, &file_paths](const boost::filesystem::path &path)
				{
					BOOST_FOREACH(const boost::filesystem::path& p, std::make_pair(boost::filesystem::recursive_directory_iterator(path),
						boost::filesystem::recursive_directory_iterator()))
					{
						if (!boost::filesystem::is_directory(p))
						{
							tstring ext(getTString(p.extension()));
#ifdef UNICODE
							std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
#else
							std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
#endif

							if (std::find(extList.begin(), extList.end(), ext) != extList.end())
							{
								const auto out_relative = relativePath(p, input_path);
								const auto out_absolute = output_path / out_relative;

								const auto out = getTString(out_absolute.branch_path() / out_absolute.stem()) + outputExt;

								file_paths.emplace_back(getTString(p), out);
							}
						}
					}

					return true;
				};

				if (!func(input_path))
					return;

				for (const auto &p : file_paths)
				{
					const boost::filesystem::path out_path(p.second);
					const boost::filesystem::path out_dir(out_path.parent_path());

					if (!boost::filesystem::exists(out_dir))
					{
						if (!boost::filesystem::create_directories(out_dir))
						{
							SendMessage(dh, WM_FAILD_CREATE_DIR, (WPARAM)&out_dir, 0);
							PostMessage(dh, WM_END_THREAD, 0, 0);
							//printf("出力フォルダ「%s」の作成に失敗しました\n", out_absolute.string().c_str());
							return;
						}
					}
				}
			}
			else
			{
				const boost::filesystem::path output_path(output_str);
				const auto outDir = output_path.branch_path();

				if (!boost::filesystem::exists(outDir))
				{
					if (!boost::filesystem::create_directories(outDir))
					{
						SendMessage(dh, WM_FAILD_CREATE_DIR, (WPARAM)&outDir, 0);
						PostMessage(dh, WM_END_THREAD, 0, 0);
						// printf("出力フォルダ「%s」の作成に失敗しました\n", output_path.string().c_str());
						return;
					}
				}

				file_paths.emplace_back(input_str, output_str);
			}
		};

		const auto inputFuncMulti = [this, &file_paths](const tstring &input)
		{
			const boost::filesystem::path input_path(boost::filesystem::absolute(input));
			const boost::filesystem::path output_path(boost::filesystem::absolute(output_str));

			const auto outilenameFunc = [&output_path](const tstring &path) -> std::wstring
			{
				const auto out = output_path / path;
				return out.wstring();
			};

			if (boost::filesystem::is_directory(input_path)) // input_pathがフォルダならそのディレクトリ以下の画像ファイルを一括変換
			{
				if (!boost::filesystem::exists(output_path))
				{
					if (!boost::filesystem::create_directory(output_path))
					{
						SendMessage(dh, WM_FAILD_CREATE_DIR, (WPARAM)&output_path, 0);
						PostMessage(dh, WM_END_THREAD, 0, 0);
						// printf("出力フォルダ「%s」の作成に失敗しました\n", output_path.string().c_str());
						return;
					}
				}

				const auto inputDirName = input_path.filename();

				// 変換する画像の入力、出力パスを取得
				const auto func = [this, &input_path, &output_path, &file_paths, &inputDirName](const boost::filesystem::path &path)
				{
					BOOST_FOREACH(const boost::filesystem::path& p, std::make_pair(boost::filesystem::recursive_directory_iterator(path),
						boost::filesystem::recursive_directory_iterator()))
					{
						if (!boost::filesystem::is_directory(p))
						{
							tstring ext(getTString(p.extension()));
#ifdef UNICODE
							std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
#else
							std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
#endif

							if (std::find(extList.begin(), extList.end(), ext) != extList.end())
							{
								const auto out_relative = relativePath(p, input_path);
								const auto out_absolute = output_path / inputDirName / out_relative;

								const auto out = getTString(out_absolute.branch_path() / out_absolute.stem()) + outputExt;

								file_paths.emplace_back(getTString(p), out);
							}
						}
					}

					return true;
				};

				if (!func(input_path))
					return;

				for (const auto &p : file_paths)
				{
					const boost::filesystem::path out_path(p.second);
					const boost::filesystem::path out_dir(out_path.parent_path());

					if (!boost::filesystem::exists(out_dir))
					{
						if (!boost::filesystem::create_directories(out_dir))
						{
							SendMessage(dh, WM_FAILD_CREATE_DIR, (WPARAM)&out_dir, 0);
							PostMessage(dh, WM_END_THREAD, 0, 0);
							//printf("出力フォルダ「%s」の作成に失敗しました\n", out_absolute.string().c_str());
							return;
						}
					}
				}
			}
			else
			{
				const auto &outDir = output_path;

				if (!boost::filesystem::exists(outDir))
				{
					if (!boost::filesystem::create_directories(outDir))
					{
						SendMessage(dh, WM_FAILD_CREATE_DIR, (WPARAM)&outDir, 0);
						PostMessage(dh, WM_END_THREAD, 0, 0);
						// printf("出力フォルダ「%s」の作成に失敗しました\n", output_path.string().c_str());
						return;
					}
				}

				const auto out = output_path / (input_path.stem().wstring() + outputExt);
				file_paths.emplace_back(input_path.wstring(), out.wstring());
			}
		};

		if(input_str_multi.size() == 0)
			inputFunc(input_str);
		else
		{
			for (const auto &path : input_str_multi)
				inputFuncMulti(path);
		}

		bool isFirst = true;

		const auto ProgessFunc = [this, &isFirst](const int ProgressFileMax, const int ProgressFileNow)
		{
			if (isFirst)
			{
				isFirst = true;

				SendMessage(GetDlgItem(dh, IDC_PROGRESS), PBM_SETRANGE32, 0, ProgressFileMax);
			}

			SendMessage(GetDlgItem(dh, IDC_PROGRESS), PBM_SETPOS, ProgressFileNow, 0);
		};

		const auto cuDNNCheckStartTime = std::chrono::system_clock::now();

		if (process == "gpu")
			Waifu2x::can_use_cuDNN();

		const auto cuDNNCheckEndTime = std::chrono::system_clock::now();

		Waifu2x::eWaifu2xError ret;

		boost::optional<double> ScaleRatio;
		boost::optional<int> ScaleWidth;
		boost::optional<int> ScaleHeight;

		switch (scaleType)
		{
		case eScaleTypeRatio:
			ScaleRatio = scale_ratio;
			break;
		case eScaleTypeWidth:
			ScaleWidth = scale_width;
			break;
		default:
			ScaleHeight = scale_height;
			break;
		}

		Waifu2x w;
		ret = w.init(__argc, __argv, mode, noise_level, ScaleRatio, ScaleWidth, ScaleHeight, model_dir, process, output_quality, output_depth, use_tta, crop_size, batch_size);
		if(ret != Waifu2x::eWaifu2xError_OK)
			SendMessage(dh, WM_ON_WAIFU2X_ERROR, (WPARAM)&ret, 0);
		else
		{
			const auto InitEndTime = std::chrono::system_clock::now();

			const int maxFile = file_paths.size();
			int num = 0;

			ProgessFunc(maxFile, 0);

			DWORD startTime = 0;

			int64_t processeNum = 0;
			int64_t count = 0;
			const auto fileNum = file_paths.size();
			for (const auto &p : file_paths)
			{
				ret = w.waifu2x(p.first, p.second, [this]()
				{
					return cancelFlag;
				});

				num++;
				ProgessFunc(maxFile, num);

				count++;

				if (ret != Waifu2x::eWaifu2xError_OK)
				{
					SendMessage(dh, WM_ON_WAIFU2X_ERROR, (WPARAM)&ret, (LPARAM)&p);

					if (ret == Waifu2x::eWaifu2xError_Cancel)
						break;
				}
				else if(count >= 2)
					processeNum++;

				if(count == 1) // 最初の一回目は二回目以降より遅くなるはずなので残り時間の計算には使わない
					startTime = timeGetTime();
				if (count >= 2)
				{
					const auto nt = timeGetTime();
					TimeLeftGetTimeThread = nt;

					const auto ElapsedTimeMS = nt - startTime;

					const double avgProcessTime = (double)ElapsedTimeMS / (double)processeNum / 1000.0;

					const auto leftnum = fileNum - count;

					const auto TimeLeft = avgProcessTime * leftnum;

					TimeLeftThread = ceil(TimeLeft);
				}
			}

			const auto ProcessEndTime = std::chrono::system_clock::now();

			cuDNNCheckTime = cuDNNCheckEndTime - cuDNNCheckStartTime;
			InitTime = InitEndTime - cuDNNCheckEndTime;
			ProcessTime = ProcessEndTime - InitEndTime;
			usedProcess = w.used_process();
		}

		PostMessage(dh, WM_END_THREAD, 0, 0);
	}

	void ReplaceAddString() // ファイル名の自動設定部分を書き換える
	{
		SyncMember(true, true);

		const boost::filesystem::path output_path(output_str);
		tstring stem;

		if (input_str_multi.size() == 0 && !boost::filesystem::is_directory(input_str))
			stem = getTString(output_path.stem());
		else
			stem = getTString(output_path.filename());

		if (stem.length() > 0 && stem.length() >= autoSetAddName.length())
		{
			const auto pos = stem.rfind(autoSetAddName);
			if (pos != tstring::npos)
			{
				const tstring addstr(AddName());

				auto new_name = stem;
				new_name.replace(pos, autoSetAddName.length(), addstr);

				autoSetAddName = addstr;

				boost::filesystem::path new_out_path;
				if (input_str_multi.size() == 0 && !boost::filesystem::is_directory(input_str))
					new_out_path = output_path.branch_path() / (new_name + outputExt);
				else
					new_out_path = output_path.branch_path() / (new_name);

				SetWindowText(GetDlgItem(dh, IDC_EDIT_OUTPUT), getTString(new_out_path).c_str());
			}
		}
	}

	void AddLogMessage(const TCHAR *msg)
	{
		if (logMessage.length() == 0)
			logMessage += msg;
		else
			logMessage += tstring(TEXT("\r\n")) + msg;

		SetWindowText(GetDlgItem(dh, IDC_EDIT_LOG), logMessage.c_str());
	}

	void Waifu2xTime()
	{
		TCHAR msg[1024 * 2];
		TCHAR *ptr = msg;

		{
			tstring p;
			if (usedProcess == "cpu")
				p = TEXT("CPU");
			else if (usedProcess == "gpu")
				p = TEXT("CUDA");
			else // if (p == "cudnn")
				p = TEXT("cuDNN");

			ptr += _stprintf(ptr, (langStringList.GetString(L"MessageUseProcessorMode") + L"\r\n").c_str(), p.c_str());
		}

		{
			uint64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(ProcessTime).count();
			const int msec = t % 1000; t /= 1000;
			const int sec = t % 60; t /= 60;
			const int min = t % 60; t /= 60;
			const int hour = (int)t;
			ptr += _stprintf(ptr, (langStringList.GetString(L"MessageProcessTime") + L"\r\n").c_str(), hour, min, sec, msec);
		}

		{
			uint64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(InitTime).count();
			const int msec = t % 1000; t /= 1000;
			const int sec = t % 60; t /= 60;
			const int min = t % 60; t /= 60;
			const int hour = (int)t;
			ptr += _stprintf(ptr, (langStringList.GetString(L"MessageInitTime") + L"\r\n").c_str(), hour, min, sec, msec);
		}

		if (process == "gpu" || process == "cudnn")
		{
			uint64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(cuDNNCheckTime).count();
			const int msec = t % 1000; t /= 1000;
			const int sec = t % 60; t /= 60;
			const int min = t % 60; t /= 60;
			const int hour = (int)t;
			ptr += _stprintf(ptr, langStringList.GetString(L"MessagecuDNNCheckTime").c_str(), hour, min, sec, msec);
		}

		AddLogMessage(msg);
	}

	void SaveIni(const bool isSyncMember = true)
	{
		if (isSyncMember)
			SyncMember(true);

		const boost::filesystem::path SettingFilePath(exeDir / SettingFileName);

		tstring tScaleRatio;
		tstring tScaleWidth;
		tstring tScaleHeight;
		tstring tmode;
		tstring tScaleMode;
		tstring tprcess;
		tstring tAutoMode;

		if (scale_ratio > 0.0)
			tScaleRatio = to_tstring(scale_ratio);
		else
			tScaleRatio = TEXT("");

		if(scale_width > 0)
			tScaleWidth = to_tstring(scale_width);
		else
			tScaleWidth = TEXT("");

		if(scale_height > 0)
			tScaleHeight = to_tstring(scale_height);
		else
			tScaleHeight = TEXT("");

		if (mode == ("noise"))
			tmode = TEXT("noise");
		else if (mode == ("scale"))
			tmode = TEXT("scale");
		else if (mode == ("auto_scale"))
			tmode = TEXT("auto_scale");
		else // noise_scale
			tmode = TEXT("noise_scale");

		if (process == "gpu")
			tprcess = TEXT("gpu");
		else
			tprcess = TEXT("cpu");

		if (scaleType == eScaleTypeRatio)
			tScaleMode = TEXT("Ratio");
		else if (scaleType == eScaleTypeWidth)
			tScaleMode = TEXT("Width");
		else
			tScaleMode = TEXT("Height");

		if (SendMessage(GetDlgItem(dh, IDC_RADIO_AUTO_START_ONE), BM_GETCHECK, 0, 0))
			tAutoMode = TEXT("one");
		else if (SendMessage(GetDlgItem(dh, IDC_RADIO_AUTO_START_MULTI), BM_GETCHECK, 0, 0))
			tAutoMode = TEXT("multi");
		else
			tAutoMode = TEXT("none");

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastScaleMode"), tScaleMode.c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastScale"), tScaleRatio.c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastScaleWidth"), tScaleWidth.c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastScaleHeight"), tScaleHeight.c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastOutputExt"), outputExt.c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastInputFileExt"), inputFileExt.c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastMode"), tmode.c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastNoiseLevel"), to_tstring(noise_level).c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastProcess"), tprcess.c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastModel"), to_tstring(modelType).c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastUseTTA"), to_tstring(use_tta ? 1 : 0).c_str(), getTString(SettingFilePath).c_str());

		if(output_quality)
			WritePrivateProfileString(TEXT("Setting"), TEXT("LastOutputQuality"), boost::lexical_cast<tstring>(*output_quality).c_str(), getTString(SettingFilePath).c_str());
		else
			WritePrivateProfileString(TEXT("Setting"), TEXT("LastOutputQuality"), TEXT(""), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastOutputDepth"), boost::lexical_cast<tstring>(output_depth).c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastLanguage"), LangName.c_str(), getTString(SettingFilePath).c_str());

		WritePrivateProfileString(TEXT("Setting"), TEXT("LastAutoMode"), tAutoMode.c_str(), getTString(SettingFilePath).c_str());
	}

	// 出力パスを選択する
	static UINT_PTR CALLBACK OFNHookProcIn(
		_In_  HWND hdlg,
		_In_  UINT uiMsg,
		_In_  WPARAM wParam,
		_In_  LPARAM lParam
		)
	{
		switch (uiMsg)
		{
		case WM_INITDIALOG:
		{
			// ダイアログを中央に表示

			HWND hParent = GetParent(hdlg);

			HWND   hwndScreen;
			RECT   rectScreen = { 0 };
			hwndScreen = GetDesktopWindow();
			GetWindowRect(hwndScreen, &rectScreen);

			RECT rDialog = { 0 };
			GetWindowRect(hParent, &rDialog);
			const int Width = rDialog.right - rDialog.left + 1;
			const int Height = rDialog.bottom - rDialog.top + 1;

			int DialogPosX;
			int DialogPosY;
			DialogPosX = ((rectScreen.right - rectScreen.left + 1) / 2 - Width / 2);
			DialogPosY = ((rectScreen.bottom - rectScreen.top + 1) / 2 - Height / 2);
			SetWindowPos(hParent, NULL, DialogPosX, DialogPosY, Width, Height, SWP_NOSIZE | SWP_NOZORDER);
		}
		break;

		case WM_NOTIFY:
		{
			// ファイルとフォルダを選択できるようにする

			NMHDR *pnmh;
			OFNOTIFY *pnot;

			pnot = (OFNOTIFY *)lParam;
			pnmh = &pnot->hdr;
			switch (pnmh->code)
			{
			case CDN_SELCHANGE:
			{
				TCHAR szPath[AR_PATH_MAX] = TEXT("");
				HWND hParent = GetParent(hdlg);
				if (CommDlg_OpenSave_GetFilePath(hParent, szPath, _countof(szPath)) > 0)
				{
					szPath[_countof(szPath) - 1] = TEXT('\0');

					boost::filesystem::path p(szPath);
					const auto filename = getTString(p.filename());

					CommDlg_OpenSave_SetControlText(hParent, edt1, filename.c_str());
				}
			}
			break;
			}
		}
		break;
		}

		return 0L;
	}

public:
	DialogEvent() : dh(nullptr), mode("noise_scale"), noise_level(1), scale_ratio(2.0), scale_width(0), scale_height(0), model_dir(TEXT("models/anime_style_art_rgb")),
		process("gpu"), outputExt(TEXT(".png")), inputFileExt(TEXT("png:jpg:jpeg:tif:tiff:bmp:tga")),
		use_tta(false), output_depth(8), crop_size(128), batch_size(1), isLastError(false), scaleType(eScaleTypeEnd),
		TimeLeftThread(-1), TimeLeftGetTimeThread(0), isCommandLineStart(false)
	{
	}

	void Exec(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		if (processThread.joinable())
			return;

		if (!SyncMember(false))
			return;

		if (input_str.length() == 0 && input_str_multi.size() == 0)
		{
			MessageBox(dh, langStringList.GetString(L"MessageInputPathCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
			return;
		}

		if (output_str.length() == 0)
		{
			MessageBox(dh, langStringList.GetString(L"MessageOutputPathCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
			return;
		}

		if (outputExt.length() == 0)
		{
			MessageBox(dh, langStringList.GetString(L"MessageOutputExtCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
			return;
		}

		if (process == "gpu" || process == "cudnn")
		{
			const auto flag = Waifu2x::can_use_CUDA();
			switch (flag)
			{
			case Waifu2x::eWaifu2xCudaError_NotFind:
				MessageBox(dh, langStringList.GetString(L"MessageCudaNotFindError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
				return;
			case Waifu2x::eWaifu2xCudaError_OldVersion:
				MessageBox(dh, langStringList.GetString(L"MessageCudaOldVersionError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
				return;
			}
		}

		SaveIni(true); // 強制終了の可能性も考えて実行時に設定保存

		SendMessage(GetDlgItem(dh, IDC_PROGRESS), PBM_SETPOS, 0, 0);
		cancelFlag = false;
		isLastError = false;

		TimeLeftThread = -1;
		TimeLeftGetTimeThread = 0;

		processThread = std::thread(std::bind(&DialogEvent::ProcessWaifu2x, this));

		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CANCEL), TRUE);
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_EXEC), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CHECK_CUDNN), FALSE);

		SetWindowText(GetDlgItem(hWnd, IDC_EDIT_LOG), TEXT(""));
		logMessage.clear();

		SetTimer(dh, nIDEventTimeLeft, 1000, NULL);
	}

	void WaitThreadExit(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		KillTimer(dh, nIDEventTimeLeft);

		processThread.join();
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CANCEL), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_EXEC), TRUE);
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CHECK_CUDNN), TRUE);

		bool endFlag = false;
		if (!isLastError)
		{
			if (!cancelFlag)
			{
				AddLogMessage(langStringList.GetString(L"MessageTransSuccess").c_str());

				if (isCommandLineStart) // コマンドライン引数を渡されて起動して、変換に成功したら終了する
					endFlag = true;
			}

			Waifu2xTime();
			MessageBeep(MB_ICONASTERISK);
		}
		else
			MessageBox(dh, langStringList.GetString(L"MessageErrorHappen").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);

		isCommandLineStart = false;

		if (endFlag)
			SendMessage(dh, WM_CLOSE, 0, 0);
	}

	void Timer(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		const int64_t TimeLeft = TimeLeftThread;
		const DWORD TimeLeftGetTime = TimeLeftGetTimeThread;

		if (TimeLeft == -1)
		{
			SetWindowText(GetDlgItem(dh, IDC_EDIT_LOG), langStringList.GetString(L"MessageTimeLeftUnkown").c_str());
		}
		else
		{
			if (TimeLeftGetTime > 0)
			{
				const DWORD tnow = timeGetTime();

				const DWORD leftprevSec = (tnow - TimeLeftGetTime) / 1000;

				int64_t TimeLeftNow = TimeLeft - (int64_t)leftprevSec;
				if (TimeLeftNow < 0)
					TimeLeftNow = 0;

				const int64_t sec = TimeLeftNow % 60;
				const int64_t min = (TimeLeftNow / 60) % 60;
				const int64_t hour = (TimeLeftNow / 60 / 60);

				TCHAR msg[1024];
				_stprintf_s(msg, TEXT("%s: %02d:%02d:%02d"), langStringList.GetString(L"MessageTimeLeft").c_str(), hour, min, sec);
				msg[_countof(msg) - 1] = TEXT('\0');

				// 表示
				SetWindowText(GetDlgItem(dh, IDC_EDIT_LOG), msg);
			}
		}
	}

	void OnDialogEnd(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		SaveIni();

		if (!processThread.joinable())
			PostQuitMessage(0);
		else
			MessageBeep(MB_ICONEXCLAMATION);
	}

	void OnFaildCreateDir(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		const boost::filesystem::path *p = (const boost::filesystem::path *)wParam;

		TCHAR msg[1024 * 2];
		_stprintf(msg, langStringList.GetString(L"MessageCreateOutDirError").c_str(), getTString(*p).c_str());

		MessageBox(dh, msg, langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);

		isLastError = true;
	}

	void OnWaifu2xError(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		const Waifu2x::eWaifu2xError ret = *(const Waifu2x::eWaifu2xError *)wParam;

		if (ret != Waifu2x::eWaifu2xError_OK)
		{
			TCHAR msg[1024] = TEXT("");

			if (lParam == 0)
			{
				switch (ret)
				{
				case Waifu2x::eWaifu2xError_Cancel:
					_stprintf(msg, langStringList.GetString(L"MessageCancelError").c_str());
					break;
				case Waifu2x::eWaifu2xError_InvalidParameter:
					_stprintf(msg, langStringList.GetString(L"MessageInvalidParameterError").c_str());
					break;
				case Waifu2x::eWaifu2xError_FailedOpenModelFile:
					_stprintf(msg, langStringList.GetString(L"MessageFailedOpenModelFileError").c_str());
					break;
				case Waifu2x::eWaifu2xError_FailedParseModelFile:
					_stprintf(msg, langStringList.GetString(L"MessageFailedParseModelFileError").c_str());
					break;
				case Waifu2x::eWaifu2xError_FailedConstructModel:
					_stprintf(msg, langStringList.GetString(L"MessageFailedConstructModelError").c_str());
					break;
				}
			}
			else
			{
				const auto &fp = *(const std::pair<tstring, tstring> *)lParam;

				switch (ret)
				{
				case Waifu2x::eWaifu2xError_Cancel:
					_stprintf(msg, langStringList.GetString(L"MessageCancelError").c_str());
					break;
				case Waifu2x::eWaifu2xError_InvalidParameter:
					_stprintf(msg, langStringList.GetString(L"MessageInvalidParameterError").c_str());
					break;
				case Waifu2x::eWaifu2xError_FailedOpenInputFile:
					_stprintf(msg, langStringList.GetString(L"MessageFailedOpenInputFileError").c_str(), fp.first.c_str());
					break;
				case Waifu2x::eWaifu2xError_FailedOpenOutputFile:
					_stprintf(msg, langStringList.GetString(L"MessageFailedOpenOutputFileError").c_str(), fp.second.c_str());
					break;
				case Waifu2x::eWaifu2xError_FailedProcessCaffe:
					_stprintf(msg, langStringList.GetString(L"MessageFailedProcessCaffeError").c_str());
					break;
				}
			}

			AddLogMessage(msg);

			if (ret != Waifu2x::eWaifu2xError_Cancel)
				isLastError = true;
		}
	}

	void SetWindowTextLang()
	{
#define SET_WINDOW_TEXT(id) SetWindowTextW(GetDlgItem(dh, id), langStringList.GetString(L#id).c_str());

		SET_WINDOW_TEXT(IDC_STATIC_IO_SETTING);
		SET_WINDOW_TEXT(IDC_STATIC_INPUT_PATH);
		SET_WINDOW_TEXT(IDC_BUTTON_INPUT_REF);
		SET_WINDOW_TEXT(IDC_STATIC_OUTPUT_PATH);
		SET_WINDOW_TEXT(IDC_STATIC_TANS_EXT_LIST);
		SET_WINDOW_TEXT(IDC_STATIC_OUTPUT_EXT);
		SET_WINDOW_TEXT(IDC_STATIC_OUTPUT_QUALITY);
		SET_WINDOW_TEXT(IDC_STATIC_OUTPUT_DEPTH);
		SET_WINDOW_TEXT(IDC_STATIC_QUALITY_PROCESS_SETTING);
		SET_WINDOW_TEXT(IDC_STATIC_TRANS_MODE);
		SET_WINDOW_TEXT(IDC_RADIO_MODE_NOISE_SCALE);
		SET_WINDOW_TEXT(IDC_RADIO_MODE_SCALE);
		SET_WINDOW_TEXT(IDC_RADIO_MODE_NOISE);
		SET_WINDOW_TEXT(IDC_RADIO_AUTO_SCALE);
		SET_WINDOW_TEXT(IDC_STATIC_JPEG_NOISE_LEVEL);
		SET_WINDOW_TEXT(IDC_RADIONOISE_LEVEL1);
		SET_WINDOW_TEXT(IDC_RADIONOISE_LEVEL2);
		SET_WINDOW_TEXT(IDC_STATIC_SCALE_RATE);
		SET_WINDOW_TEXT(IDC_RADIO_SCALE_RATIO);
		SET_WINDOW_TEXT(IDC_RADIO_SCALE_WIDTH);
		SET_WINDOW_TEXT(IDC_RADIO_SCALE_HEIGHT);
		SET_WINDOW_TEXT(IDC_STATIC_MODEL);
		SET_WINDOW_TEXT(IDC_RADIO_MODEL_RGB);
		SET_WINDOW_TEXT(IDC_RADIO_MODEL_PHOTO);
		SET_WINDOW_TEXT(IDC_RADIO_MODEL_Y);
		SET_WINDOW_TEXT(IDC_CHECK_TTA);
		SET_WINDOW_TEXT(IDC_STATIC_PROCESS_SPEED_SETTING);
		SET_WINDOW_TEXT(IDC_STATIC_PROCESSOR);
		SET_WINDOW_TEXT(IDC_RADIO_MODE_GPU);
		SET_WINDOW_TEXT(IDC_RADIO_MODE_CPU);
		SET_WINDOW_TEXT(IDC_STATIC_CROP_SIZE);
		SET_WINDOW_TEXT(IDC_BUTTON_CHECK_CUDNN);
		SET_WINDOW_TEXT(IDC_BUTTON_CANCEL);
		SET_WINDOW_TEXT(IDC_BUTTON_EXEC);
		SET_WINDOW_TEXT(IDC_STATIC_LANG_UI);
		SET_WINDOW_TEXT(IDC_STATIC_AUTO_START);
		SET_WINDOW_TEXT(IDC_RADIO_AUTO_START_NONE);
		SET_WINDOW_TEXT(IDC_RADIO_AUTO_START_ONE);
		SET_WINDOW_TEXT(IDC_RADIO_AUTO_START_MULTI);

#undef SET_WINDOW_TEXT
	}

	void SetDepthAndQuality(const bool SetDefaultQuality = true)
	{
		HWND hout = GetDlgItem(dh, IDC_COMBO_OUT_EXT);
		HWND houtDepth = GetDlgItem(dh, IDC_COMBO_OUTPUT_DEPTH);

		const int cur = SendMessage(hout, CB_GETCURSEL, 0, 0);
		if (cur < 0)
			return;

		const auto &OutputExtentionList = Waifu2x::OutputExtentionList;
		if (cur >= OutputExtentionList.size())
			return;

		const auto elm = OutputExtentionList[cur];

		int oldDepth = 0;
		{
			TCHAR oldDepthStr[100] = TEXT("");
			GetWindowText(houtDepth, oldDepthStr, _countof(oldDepthStr));

			if (_tcslen(oldDepthStr) > 0)
				oldDepth = boost::lexical_cast<int>(oldDepthStr);
		}

		// 深度のリスト初期化
		while (SendMessage(houtDepth, CB_GETCOUNT, 0, 0) != 0)
			SendMessage(houtDepth, CB_DELETESTRING, 0, 0);

		// 深度のリスト追加
		size_t defaultIndex = 0;
		for (size_t i = 0; i < elm.depthList.size(); i++)
		{
			const auto depth = elm.depthList[i];

			const auto str = boost::lexical_cast<tstring>(depth);
			const auto index = SendMessage(houtDepth, CB_ADDSTRING, 0, (LPARAM)str.c_str());

			if (depth == oldDepth)
				defaultIndex = i;
		}

		SendMessage(houtDepth, CB_SETCURSEL, defaultIndex, 0);

		if (elm.depthList.size() == 1)
			EnableWindow(houtDepth, FALSE);
		else
			EnableWindow(houtDepth, TRUE);

		if (!elm.imageQualityStart || !elm.imageQualityEnd || !elm.imageQualityDefault) // 画質設定は無効
		{
			EnableWindow(GetDlgItem(dh, IDC_EDIT_OUT_QUALITY), FALSE);
			SetWindowTextW(GetDlgItem(dh, IDC_EDIT_OUT_QUALITY), L"");

			SetWindowTextW(GetDlgItem(dh, IDC_STATIC_OUTPUT_QUALITY), langStringList.GetString(L"IDC_STATIC_OUTPUT_QUALITY").c_str());

			output_quality.reset();
		}
		else
		{
			HWND hedit = GetDlgItem(dh, IDC_EDIT_OUT_QUALITY);

			EnableWindow(hedit, TRUE);
			if(SetDefaultQuality)
				SetWindowText(hedit, boost::lexical_cast<tstring>(*elm.imageQualityDefault).c_str());

			const auto wstr = langStringList.GetString(L"IDC_STATIC_OUTPUT_QUALITY");

			const auto addstr = std::wstring(L" (") + boost::lexical_cast<std::wstring>(*elm.imageQualityStart)
				+ L"～" + boost::lexical_cast<std::wstring>(*elm.imageQualityEnd) + L")";
			SetWindowTextW(GetDlgItem(dh, IDC_STATIC_OUTPUT_QUALITY), (wstr + addstr).c_str());
		}
	}

	void Create(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		dh = hWnd;

		{
			TCHAR texepath[1024 * 3] = TEXT("");
			GetModuleFileName(NULL, texepath, _countof(texepath));
			texepath[_countof(texepath) - 1] = TEXT('\0');

			const boost::filesystem::path exePath(texepath);
			exeDir = exePath.branch_path();
		}

		const boost::filesystem::path SettingFilePath(exeDir / SettingFileName);

		{
			const boost::filesystem::path LangDirPath(exeDir / LangDir);
			const boost::filesystem::path LangListPath(exeDir / LangListFileName);
			langStringList.SetLangBaseDir(getTString(LangDirPath));
			langStringList.ReadLangList(getTString(LangListPath));
		}

		std::wstring langName;
		{
			TCHAR tmp[1000];

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastLanguage"), TEXT(""), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			langName = tmp;
		}

		{
			HWND hlang = GetDlgItem(dh, IDC_COMBO_LANG);

			const auto &list = langStringList.GetLangList();
			for (const auto &lang : list)
			{
				const int index = SendMessageW(hlang, CB_ADDSTRING, 0, (LPARAM)lang.LangName.c_str());
			}

			size_t defaultListIndex = 0;
			const auto &DefaultLang = langStringList.GetLang();
			for (size_t i = 0; i < list.size(); i++)
			{
				const auto &lang = list[i];
				if (lang.LangName == DefaultLang.LangName && lang.LangID == DefaultLang.LangID
					&& lang.SubLangID == DefaultLang.SubLangID && lang.FileName == DefaultLang.FileName)
				{
					defaultListIndex = i;
					LangName = list[i].LangName;
					break;
				}
			}

			if (langName.length() > 0) // 前回起動時の言語があったらそっちを優先
			{
				for (size_t i = 0; i < list.size(); i++)
				{
					const auto &lang = list[i];
					if (lang.LangName == langName)
					{
						defaultListIndex = i;
						langStringList.SetLang(list[i]);

						LangName = langName;
						break;
					}
				}
			}

			SendMessage(hlang, CB_SETCURSEL, defaultListIndex, 0);
		}

		SetWindowTextLang();

		{
			HWND houtext = GetDlgItem(dh, IDC_COMBO_OUT_EXT);

			const auto &OutputExtentionList = Waifu2x::OutputExtentionList;
			for (const auto &elm : OutputExtentionList)
			{
				SendMessageW(houtext, CB_ADDSTRING, 0, (LPARAM)elm.ext.c_str());
			}

			SendMessage(houtext, CB_SETCURSEL, 0, 0);
		}

		const boost::filesystem::path CropSizeListPath(exeDir / CropSizeListName);
		std::ifstream ifs(CropSizeListPath.wstring());
		if (ifs)
		{
			std::string str;
			while (getline(ifs, str))
			{
				char *ptr = nullptr;
				const long n = strtol(str.c_str(), &ptr, 10);
				if (ptr && *ptr == '\0')
					CropSizeList.push_back(n);
			}
		}

		{
			HWND hcrop = GetDlgItem(dh, IDC_COMBO_CROP_SIZE);

			SendMessage(hcrop, CB_ADDSTRING, 0, (LPARAM)TEXT("-----------------------"));

			// CropSizeListの値を追加していく
			int mindiff = INT_MAX;
			int defaultListIndex = -1;
			for (const auto n : CropSizeList)
			{
				tstring str(to_tstring(n));
				const int index = SendMessage(hcrop, CB_ADDSTRING, 0, (LPARAM)str.c_str());

				const int diff = abs(DefaultCommonDivisor - n);
				if (DefaultCommonDivisorRange.first <= n && n <= DefaultCommonDivisorRange.second && diff < mindiff)
				{
					mindiff = diff;
					defaultListIndex = index;
				}
			}

			if (GetWindowTextLength(hcrop) == 0)
				SendMessage(hcrop, CB_SETCURSEL, defaultListIndex, 0);
		}

		tstring tScaleRatio;
		tstring tScaleWidth;
		tstring tScaleHeight;

		tstring tScaleMode;
		tstring tmode;
		tstring tprcess;
		tstring tAutoMode;
		{
			TCHAR tmp[1000];

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastScale"), TEXT("2.00"), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			tScaleRatio = tmp;

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastScaleWidth"), TEXT("0"), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			tScaleWidth = tmp;

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastScaleHeight"), TEXT("0"), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			tScaleHeight = tmp;

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastScaleMode"), TEXT("Ratio"), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			tScaleMode = tmp;

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastOutputExt"), TEXT("png"), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			outputExt = tmp;

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastInputFileExt"), TEXT("png:jpg:jpeg:tif:tiff:bmp:tga"), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			inputFileExt = tmp;

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastMode"), TEXT("noise_scale"), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			tmode = tmp;

			noise_level = GetPrivateProfileInt(TEXT("Setting"), TEXT("LastNoiseLevel"), 1, getTString(SettingFilePath).c_str());

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastProcess"), TEXT("gpu"), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			tprcess = tmp;

			modelType = (eModelType)GetPrivateProfileInt(TEXT("Setting"), TEXT("LastModel"), 0, getTString(SettingFilePath).c_str());

			use_tta = GetPrivateProfileInt(TEXT("Setting"), TEXT("LastUseTTA"), 0, getTString(SettingFilePath).c_str()) != 0;

			output_quality.reset();
			const int num = GetPrivateProfileInt(TEXT("Setting"), TEXT("LastOutputQuality"), -100, getTString(SettingFilePath).c_str());
			if (num != -100)
				output_quality = num;

			output_depth = GetPrivateProfileInt(TEXT("Setting"), TEXT("LastOutputDepth"), output_depth, getTString(SettingFilePath).c_str());

			GetPrivateProfileString(TEXT("Setting"), TEXT("LastAutoMode"), TEXT("none"), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
			tmp[_countof(tmp) - 1] = TEXT('\0');
			tAutoMode = tmp;
		}

		TCHAR *ptr = nullptr;
		const double tempScale = _tcstod(tScaleRatio.c_str(), &ptr);
		if (!ptr || *ptr != TEXT('\0') || tempScale <= 0.0)
			tScaleRatio = TEXT("2.00");

		const long tempScaleWidth = _tcstol(tScaleWidth.c_str(), &ptr, 10);
		if (!ptr || *ptr != TEXT('\0') || tempScaleWidth <= 0)
			tScaleWidth = TEXT("");

		const long tempScaleHeight = _tcstol(tScaleHeight.c_str(), &ptr, 10);
		if (!ptr || *ptr != TEXT('\0') || tempScaleHeight <= 0)
			tScaleHeight = TEXT("");

		if (outputExt.length() > 0 && outputExt[0] != TEXT('.'))
			outputExt = L"." + outputExt;

		if (!(1 <= noise_level && noise_level <= 2))
			noise_level = 1;

		if (tprcess == TEXT("gpu"))
			process = "gpu";
		else
			process = "cpu";

		if (!((eModelType)0 <= modelType && modelType < eModelTypeEnd))
			modelType = eModelTypeRGB;

		if (tmode == TEXT("noise"))
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_NOISE_SCALE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_SCALE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_NOISE), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_SCALE), BM_SETCHECK, BST_UNCHECKED, 0);
		}
		else if (tmode == TEXT("scale"))
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_NOISE_SCALE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_SCALE), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_NOISE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_SCALE), BM_SETCHECK, BST_UNCHECKED, 0);
		}
		else if (tmode == TEXT("auto_scale"))
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_NOISE_SCALE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_SCALE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_NOISE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_SCALE), BM_SETCHECK, BST_CHECKED, 0);
		}
		else // noise_scale
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_NOISE_SCALE), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_SCALE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_NOISE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_SCALE), BM_SETCHECK, BST_UNCHECKED, 0);
		}

		if(tScaleMode == TEXT("Ratio"))
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_RATIO), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_WIDTH), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_HEIGHT), BM_SETCHECK, BST_UNCHECKED, 0);

			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), TRUE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), FALSE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), FALSE);
		}
		else if (tScaleMode == TEXT("Width"))
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_RATIO), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_WIDTH), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_HEIGHT), BM_SETCHECK, BST_UNCHECKED, 0);

			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), FALSE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), TRUE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), FALSE);
		}
		else
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_RATIO), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_WIDTH), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_HEIGHT), BM_SETCHECK, BST_CHECKED, 0);

			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), FALSE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), FALSE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), TRUE);
		}

		if (noise_level == 1)
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL1), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL2), BM_SETCHECK, BST_UNCHECKED, 0);
		}
		else
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL1), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL2), BM_SETCHECK, BST_CHECKED, 0);
		}
		
		if (process == "gpu")
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_GPU), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_CPU), BM_SETCHECK, BST_UNCHECKED, 0);
		}
		else
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_GPU), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_CPU), BM_SETCHECK, BST_CHECKED, 0);
		}

		if (modelType == eModelTypeRGB)
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_RGB), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_PHOTO), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_Y), BM_SETCHECK, BST_UNCHECKED, 0);
		}
		else if (modelType == eModelTypePhoto)
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_RGB), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_PHOTO), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_Y), BM_SETCHECK, BST_UNCHECKED, 0);
		}
		else
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_RGB), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_PHOTO), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_Y), BM_SETCHECK, BST_CHECKED, 0);
		}

		if (use_tta)
			SendMessage(GetDlgItem(hWnd, IDC_CHECK_TTA), BM_SETCHECK, BST_CHECKED, 0);
		else
			SendMessage(GetDlgItem(hWnd, IDC_CHECK_TTA), BM_SETCHECK, BST_UNCHECKED, 0);

		SetWindowText(GetDlgItem(hWnd, IDC_EDIT_SCALE_RATIO), tScaleRatio.c_str());
		SetWindowText(GetDlgItem(hWnd, IDC_EDIT_SCALE_WIDTH), tScaleWidth.c_str());
		SetWindowText(GetDlgItem(hWnd, IDC_EDIT_SCALE_HEIGHT), tScaleHeight.c_str());

		SetWindowText(GetDlgItem(hWnd, IDC_EDIT_INPUT_EXT_LIST), inputFileExt.c_str());

		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CANCEL), FALSE);

		// 前回の拡張子設定関連を復元
		HWND houtext = GetDlgItem(dh, IDC_COMBO_OUT_EXT);

		size_t defaultIndex = 0;
		const auto &OutputExtentionList = Waifu2x::OutputExtentionList;
		for (size_t i = 0; i < OutputExtentionList.size(); i++)
		{
			const auto &elm = OutputExtentionList[i];
			if (elm.ext == outputExt)
			{
				defaultIndex = i;
				break;
			}
		}

		SendMessage(houtext, CB_SETCURSEL, defaultIndex, 0);

		if(output_quality)
			SetWindowText(GetDlgItem(hWnd, IDC_EDIT_OUT_QUALITY), boost::lexical_cast<tstring>(*output_quality).c_str());

		SetWindowText(GetDlgItem(hWnd, IDC_COMBO_OUTPUT_DEPTH), boost::lexical_cast<tstring>(output_depth).c_str());

		if (tAutoMode == TEXT("one"))
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_START_NONE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_START_MULTI), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_START_ONE), BM_SETCHECK, BST_CHECKED, 0);
		}
		else if (tAutoMode == TEXT("multi"))
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_START_NONE), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_START_MULTI), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_START_ONE), BM_SETCHECK, BST_UNCHECKED, 0);
		}
		else
		{
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_START_NONE), BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_START_MULTI), BM_SETCHECK, BST_UNCHECKED, 0);
			SendMessage(GetDlgItem(hWnd, IDC_RADIO_AUTO_START_ONE), BM_SETCHECK, BST_UNCHECKED, 0);
		}

		SetDepthAndQuality(false);


		int nArgs = 0;
		LPTSTR *lplpszArgs;
		lplpszArgs = CommandLineToArgvW(GetCommandLine(), &nArgs);
		if (lplpszArgs)
		{
			input_str_multi.clear();

			if (nArgs > 1)
			{
				if (nArgs == 2)
				{
					OnSetInputFilePath(lplpszArgs[1]);
				}
				else if (nArgs > 2)
				{
					for (int i = 1; i < nArgs; i++)
						input_str_multi.push_back(lplpszArgs[i]);

					OnSetInputFilePath();
				}

				isCommandLineStart = true;

				::PostMessage(GetDlgItem(dh, IDC_BUTTON_EXEC), BM_CLICK, 0, 0);
			}

			LocalFree(lplpszArgs);
		}
	}

	void Cancel(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		cancelFlag = true;
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CANCEL), FALSE);
	}

	void UpdateAddString(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		ReplaceAddString();
	}

	void ScaleRadio(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		if (SendMessage(GetDlgItem(dh, IDC_RADIO_SCALE_RATIO), BM_GETCHECK, 0, 0))
		{
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), TRUE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), FALSE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), FALSE);
		}
		else if (SendMessage(GetDlgItem(dh, IDC_RADIO_SCALE_WIDTH), BM_GETCHECK, 0, 0))
		{
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), FALSE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), TRUE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), FALSE);
		}
		else
		{
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), FALSE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), FALSE);
			EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), TRUE);
		}

		ReplaceAddString();
	}

	void CheckCUDNN(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		const auto flag = Waifu2x::can_use_CUDA();
		switch (flag)
		{
		case Waifu2x::eWaifu2xCudaError_NotFind:
			MessageBox(dh, langStringList.GetString(L"MessageCudaNotFindError").c_str(), langStringList.GetString(L"MessageTitleResult").c_str(), MB_OK | MB_ICONERROR);
			return;
		case Waifu2x::eWaifu2xCudaError_OldVersion:
			MessageBox(dh, langStringList.GetString(L"MessageCudaOldVersionError").c_str(), langStringList.GetString(L"MessageTitleResult").c_str(), MB_OK | MB_ICONERROR);
			return;
		}

		switch (Waifu2x::can_use_cuDNN())
		{
		case Waifu2x::eWaifu2xcuDNNError_OK:
			MessageBox(dh, langStringList.GetString(L"MessagecuDNNOK").c_str(), langStringList.GetString(L"MessageTitleResult").c_str(), MB_OK | MB_ICONINFORMATION);
			break;
		case Waifu2x::eWaifu2xcuDNNError_NotFind:
		{
			TCHAR msg[1024 * 2];
			_stprintf(msg, langStringList.GetString(L"MessagecuDNNNotFindError").c_str(), TEXT(CUDNN_DLL_NAME));
			MessageBox(dh, msg, langStringList.GetString(L"MessageTitleResult").c_str(), MB_OK | MB_ICONERROR);
			break;
		}
		case Waifu2x::eWaifu2xcuDNNError_OldVersion:
		{
			TCHAR msg[1024 * 2];
			_stprintf(msg, langStringList.GetString(L"MessagecuDNNOldVersionError").c_str(), TEXT(CUDNN_DLL_NAME), TEXT(CUDNN_REQUIRE_VERION_TEXT));
			MessageBox(dh, msg, langStringList.GetString(L"MessageTitleResult").c_str(), MB_OK | MB_ICONERROR);
			break;
		}
		case Waifu2x::eWaifu2xcuDNNError_CannotCreate:
			MessageBox(dh, langStringList.GetString(L"MessagecuDNNCannotCreateError").c_str(), langStringList.GetString(L"MessageTitleResult").c_str(), MB_OK | MB_ICONERROR);
			break;
		default:
			MessageBox(dh, langStringList.GetString(L"MessagecuDNNDefautlError").c_str(), langStringList.GetString(L"MessageTitleResult").c_str(), MB_OK | MB_ICONERROR);
		}
	}

	LRESULT OnSetInputFilePath(const TCHAR *tPath)
	{
		HWND hWnd = GetDlgItem(dh, IDC_EDIT_INPUT);

		boost::filesystem::path path(tPath);

		if (!boost::filesystem::exists(path))
		{
			MessageBox(dh, langStringList.GetString(L"MessageInputCheckError").c_str(), langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);
			return 0L;
		}

		input_str_multi.clear();

		SyncMember(true, true);

		if (boost::filesystem::is_directory(path))
		{
			HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

			const tstring addstr(AddName());
			autoSetAddName = AddName();

			const auto str = getTString(path.branch_path() / (path.stem().wstring() + addstr));
			SetWindowText(ho, str.c_str());

			SetWindowText(hWnd, tPath);
		}
		else
		{
			HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

			tstring outputFileName = tPath;

			const auto tailDot = outputFileName.find_last_of('.');
			if (tailDot != outputFileName.npos)
				outputFileName.erase(tailDot, outputFileName.length());

			const tstring addstr(AddName());
			autoSetAddName = addstr;

			outputFileName += addstr + outputExt;

			SetWindowText(ho, outputFileName.c_str());

			SetWindowText(hWnd, tPath);
		}

		SetCropSizeList(path);

		return 0L;
	}

	LRESULT OnSetInputFilePath()
	{
		HWND hWnd = GetDlgItem(dh, IDC_EDIT_INPUT);

		SyncMember(true, true);

		if (input_str_multi.size() > 0)
		{
			SetWindowText(hWnd, MultiFileStr);

			HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

			const tstring addstr(AddName());
			autoSetAddName = AddName();

			boost::filesystem::path baseDir(input_str_multi[0]);

			tstring filename;
			if (boost::filesystem::is_directory(baseDir))
				filename = baseDir.filename().wstring();
			else
				filename = baseDir.stem().wstring();

			const auto str = getTString(baseDir.branch_path() / (filename + TEXT(" multi") + addstr));
			SetWindowText(ho, str.c_str());
		}

		SetCropSizeList(TEXT(""));

		return 0L;
	}

	// ここで渡されるhWndはIDC_EDITのHWND(コントロールのイベントだから)
	LRESULT DropInput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData)
	{
		TCHAR szTmp[AR_PATH_MAX];

		// ドロップされたファイル数を取得
		UINT FileNum = DragQueryFile((HDROP)wParam, 0xFFFFFFFF, szTmp, _countof(szTmp));
		if (FileNum > 0)
		{
			if (FileNum == 1)
			{
				DragQueryFile((HDROP)wParam, 0, szTmp, _countof(szTmp));
				szTmp[_countof(szTmp) - 1] = TEXT('\0');

				OnSetInputFilePath(szTmp);
			}
			else if (FileNum > 1)
			{
				input_str.clear();
				input_str_multi.clear();

				for (UINT i = 0; i < FileNum; i++)
				{
					TCHAR szTmp[AR_PATH_MAX];

					if (DragQueryFile((HDROP)wParam, i, szTmp, _countof(szTmp)) < _countof(szTmp))
					{
						szTmp[_countof(szTmp) - 1] = TEXT('\0');

						input_str_multi.push_back(szTmp);
					}
				}

				OnSetInputFilePath();
			}

			if (SendMessage(GetDlgItem(dh, IDC_RADIO_AUTO_START_ONE), BM_GETCHECK, 0, 0) ||
				(SendMessage(GetDlgItem(dh, IDC_RADIO_AUTO_START_MULTI), BM_GETCHECK, 0, 0) && (input_str_multi.size() > 0 || boost::filesystem::is_directory(szTmp))))
			{
				::PostMessage(GetDlgItem(dh, IDC_BUTTON_EXEC), BM_CLICK, 0, 0);
			}
		}

		return 0L;
	}

	// ここで渡されるhWndはIDC_EDITのHWND(コントロールのイベントだから)
	LRESULT DropOutput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData)
	{
		TCHAR szTmp[AR_PATH_MAX];

		// ドロップされたファイル数を取得
		UINT FileNum = DragQueryFile((HDROP)wParam, 0xFFFFFFFF, szTmp, AR_PATH_MAX);
		if (FileNum >= 1)
		{
			DragQueryFile((HDROP)wParam, 0, szTmp, AR_PATH_MAX);
			SetWindowText(hWnd, szTmp);
		}

		return 0L;
	}

	LRESULT TextInput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData)
	{
		const auto ret = CallWindowProc(OrgSubWnd, hWnd, WM_CHAR, wParam, lParam);
		ReplaceAddString();
		return ret;
	}

	void InputRef(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		SyncMember(false);

		OPENFILENAME ofn = { 0 };
		TCHAR szPath[AR_PATH_MAX] = TEXT("");

		static std::vector<TCHAR> szFile(AR_PATH_MAX * 100);
		memset(szFile.data(), 0, szFile.size() * sizeof(TCHAR));

		GetCurrentDirectory(_countof(szPath), szPath);
		szPath[_countof(szPath) - 1] = TEXT('\0');

		tstring extStr;
		for (const auto &ext : extList)
		{
			if (extStr.length() != 0)
				extStr += TEXT(";*") + ext;
			else
				extStr = TEXT("*") + ext;
		}

		TCHAR szFilter[AR_PATH_MAX] = TEXT("");
		TCHAR *tfp = szFilter;

		if (extStr.length() > 0)
		{
			tfp += _stprintf(tfp, langStringList.GetString(L"MessageExtStr").c_str(), extStr.c_str());
			tfp++;

			memcpy(tfp, extStr.c_str(), extStr.length() * sizeof(TCHAR));
			tfp += extStr.length();

			*tfp = TEXT('\0');
			tfp++;
		}

		const tstring allFilesTitle(langStringList.GetString(L"MessageAllFileFolder").c_str());
		memcpy(tfp, allFilesTitle.c_str(), allFilesTitle.length() * sizeof(TCHAR));
		tfp += allFilesTitle.length();
		*tfp = TEXT('\0');
		tfp++;

		const tstring allFilesExt(TEXT("*.*"));
		memcpy(tfp, allFilesExt.c_str(), allFilesExt.length() * sizeof(TCHAR));
		tfp += allFilesExt.length();

		*tfp = TEXT('\0');
		tfp++;
		*tfp = TEXT('\0');
		tfp++;

		ofn.lStructSize = sizeof(ofn);
		ofn.hwndOwner = dh;
		ofn.lpstrFile = szFile.data();
		ofn.nMaxFile = szFile.size();
		ofn.lpstrFilter = szFilter;
		ofn.nFilterIndex = 1;
		ofn.lpstrTitle = langStringList.GetString(L"MessageTitleInputDialog").c_str();
		ofn.lpstrInitialDir = szPath;
		ofn.lpstrCustomFilter = NULL;
		ofn.nMaxCustFilter = 0;
		ofn.lpstrFileTitle = NULL;
		ofn.nMaxFileTitle = 0;
		ofn.nFileOffset = 0;
		ofn.nFileExtension = 0;
		ofn.lpstrDefExt = NULL;
		ofn.lCustData = 0;
		ofn.lpfnHook = OFNHookProcIn;
		ofn.lpTemplateName = 0;
		//ofn.Flags = OFN_HIDEREADONLY | OFN_NOVALIDATE | OFN_PATHMUSTEXIST | OFN_READONLY | OFN_EXPLORER | OFN_ENABLEHOOK | OFN_FILEMUSTEXIST | OFN_ALLOWMULTISELECT;
		ofn.Flags = OFN_HIDEREADONLY | OFN_NOVALIDATE | OFN_PATHMUSTEXIST | OFN_ENABLESIZING | OFN_READONLY | OFN_EXPLORER | OFN_ENABLEHOOK | OFN_FILEMUSTEXIST | OFN_ALLOWMULTISELECT;
		if (GetOpenFileName(&ofn))
		{
			szFile[szFile.size() - 1] = TEXT('\0');

			const TCHAR * ptr = szFile.data();

			const auto firstLen = _tcslen(ptr);
			if (firstLen > 0)
			{
				input_str.clear();
				input_str_multi.clear();

				if(firstLen + 2 >= szFile.size() || ptr[firstLen + 1] == '\0')
					OnSetInputFilePath(ptr);
				else
				{
					const TCHAR * end = ptr + szFile.size();

					const tstring baseDir(ptr);
					ptr += firstLen + 1;

					while (ptr < end)
					{
						if (*ptr == TEXT('\0'))
							break;

						TCHAR szTmp[AR_PATH_MAX];

						const auto len = _tcslen(ptr);
						memcpy(szTmp, ptr, len * sizeof(TCHAR));
						szTmp[len] = TEXT('\0');

						const auto str = baseDir + TEXT('\\') + szTmp;

						input_str_multi.push_back(str);

						ptr += len + 1;
					}

					OnSetInputFilePath();
				}

				if ( SendMessage(GetDlgItem(dh, IDC_RADIO_AUTO_START_ONE), BM_GETCHECK, 0, 0) ||
					(SendMessage(GetDlgItem(dh, IDC_RADIO_AUTO_START_MULTI), BM_GETCHECK, 0, 0) && (input_str_multi.size() > 0 || boost::filesystem::is_directory(szFile.data()))))
				{
					::PostMessage(GetDlgItem(dh, IDC_BUTTON_EXEC), BM_CLICK, 0, 0);
				}
			}
		}
	}

	void LangChange(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		if (HIWORD(wParam) != CBN_SELCHANGE)
			return;

		HWND hlang = GetDlgItem(dh, IDC_COMBO_LANG);

		const int cur = SendMessage(hlang, CB_GETCURSEL, 0, 0);

		const auto &list = langStringList.GetLangList();

		if (list.size() <= cur)
			return;

		langStringList.SetLang(list[cur]);
		LangName = list[cur].LangName;

		SetWindowTextLang();
	}

	void OutExtChange(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		if (HIWORD(wParam) != CBN_SELCHANGE)
			return;

		SetDepthAndQuality();

		ReplaceAddString();
	}
};

int WINAPI WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR     lpCmdLine,
	int       nCmdShow)
{
	Waifu2x::init_liblary();

	// 管理者権限で起動してもファイルのドロップを受け付けるようにする
	ChangeWindowMessageFilter(WM_DROPFILES, MSGFLT_ADD);
	ChangeWindowMessageFilter(WM_COPYDATA, MSGFLT_ADD);
	ChangeWindowMessageFilter(0x0049, MSGFLT_ADD);

	// Caffeのエラーでないログを保存しないようにする
	google::SetLogDestination(google::GLOG_INFO, "");
	google::SetLogDestination(google::GLOG_WARNING, "");

	// Caffeのエラーログを「error_log_～」に出力
	google::SetLogDestination(google::GLOG_ERROR, "error_log_");
	google::SetLogDestination(google::GLOG_FATAL, "error_log_");

	// CDialogクラスでダイアログを作成する
	CDialog cDialog;
	CDialog cDialog2;
	// IDC_EDITのサブクラス
	CControl cControlInput(IDC_EDIT_INPUT);
	CControl cControlOutput(IDC_EDIT_OUTPUT);
	CControl cControlScaleRatio(IDC_EDIT_SCALE_RATIO);
	CControl cControlScaleWidth(IDC_EDIT_SCALE_WIDTH);
	CControl cControlScaleHeight(IDC_EDIT_SCALE_HEIGHT);

	// 登録する関数がまとめられたクラス
	// グローバル関数を使えばクラスにまとめる必要はないがこの方法が役立つこともあるはず
	DialogEvent cDialogEvent;

	// クラスの関数を登録する場合

	// IDC_EDITにWM_DROPFILESが送られてきたときに実行する関数の登録
	cControlInput.SetEventCallBack(SetClassCustomFunc(DialogEvent::DropInput, &cDialogEvent), NULL, WM_DROPFILES);
	cControlOutput.SetEventCallBack(SetClassCustomFunc(DialogEvent::DropOutput, &cDialogEvent), NULL, WM_DROPFILES);

	cControlScaleRatio.SetEventCallBack(SetClassCustomFunc(DialogEvent::TextInput, &cDialogEvent), NULL, WM_CHAR);
	cControlScaleWidth.SetEventCallBack(SetClassCustomFunc(DialogEvent::TextInput, &cDialogEvent), NULL, WM_CHAR);
	cControlScaleHeight.SetEventCallBack(SetClassCustomFunc(DialogEvent::TextInput, &cDialogEvent), NULL, WM_CHAR);

	// コントロールのサブクラスを登録
	cDialog.AddControl(&cControlInput);
	cDialog.AddControl(&cControlOutput);
	cDialog.AddControl(&cControlScaleRatio);
	cDialog.AddControl(&cControlScaleWidth);
	cDialog.AddControl(&cControlScaleHeight);

	// 各コントロールのイベントで実行する関数の登録
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::Exec, &cDialogEvent), NULL, IDC_BUTTON_EXEC);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::Cancel, &cDialogEvent), NULL, IDC_BUTTON_CANCEL);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::InputRef, &cDialogEvent), NULL, IDC_BUTTON_INPUT_REF);

	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIO_MODE_NOISE);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIO_MODE_SCALE);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIO_MODE_NOISE_SCALE);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIO_AUTO_SCALE);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIONOISE_LEVEL1);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIONOISE_LEVEL2);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIO_MODE_CPU);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIO_MODE_GPU);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIO_MODEL_RGB);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIO_MODEL_PHOTO);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_RADIO_MODEL_Y);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_CHECK_TTA);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::UpdateAddString, &cDialogEvent), NULL, IDC_COMBO_OUTPUT_DEPTH);

	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::ScaleRadio, &cDialogEvent), NULL, IDC_RADIO_SCALE_RATIO);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::ScaleRadio, &cDialogEvent), NULL, IDC_RADIO_SCALE_WIDTH);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::ScaleRadio, &cDialogEvent), NULL, IDC_RADIO_SCALE_HEIGHT);

	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::CheckCUDNN, &cDialogEvent), NULL, IDC_BUTTON_CHECK_CUDNN);

	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::LangChange, &cDialogEvent), NULL, IDC_COMBO_LANG);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::OutExtChange, &cDialogEvent), NULL, IDC_COMBO_OUT_EXT);

	// ダイアログのイベントで実行する関数の登録
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::Create, &cDialogEvent), NULL, WM_INITDIALOG);
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::OnDialogEnd, &cDialogEvent), NULL, WM_CLOSE);
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::OnFaildCreateDir, &cDialogEvent), NULL, WM_FAILD_CREATE_DIR);
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::OnWaifu2xError, &cDialogEvent), NULL, WM_ON_WAIFU2X_ERROR);
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::WaitThreadExit, &cDialogEvent), NULL, WM_END_THREAD);
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::Timer, &cDialogEvent), NULL, WM_TIMER);

	// ダイアログを表示
	cDialog.DoModal(hInstance, IDD_DIALOG);

	Waifu2x::quit_liblary();

	return 0;
}
