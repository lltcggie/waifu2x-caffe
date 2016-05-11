#include "MainDialog.h"
#include <Commctrl.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cblas.h>
#include <dlgs.h>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <boost/math/common_factor_rt.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../common/waifu2x.h"
#include "CDialog.h"
#include "CControl.h"
//#include <boost/program_options.hpp>
#include <tclapw/CmdLine.h>


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
	while (*itr_path == *itr_relative_to && itr_path != p.end() && itr_relative_to != r.end())
	{
		++itr_path;
		++itr_relative_to;
	}

	// add "../" for each remaining token in relative_to
	if (itr_relative_to != r.end())
	{
		++itr_relative_to;
		while (itr_relative_to != r.end())
		{
			result /= "..";
			++itr_relative_to;
		}
	}

	// add remaining path
	while (itr_path != p.end())
	{
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


tstring DialogEvent::AddName() const
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

bool DialogEvent::SyncMember(const bool NotSyncCropSize, const bool silent)
{
	bool ret = true;

	if (input_str_multi.size() == 0)
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
	else if (SendMessage(GetDlgItem(dh, IDC_RADIONOISE_LEVEL2), BM_GETCHECK, 0, 0))
		noise_level = 2;
	else
		noise_level = 3;

	if (SendMessage(GetDlgItem(dh, IDC_RADIO_SCALE_RATIO), BM_GETCHECK, 0, 0))
		scaleType = eScaleTypeRatio;
	else if (SendMessage(GetDlgItem(dh, IDC_RADIO_SCALE_WIDTH), BM_GETCHECK, 0, 0))
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

				if (!silent)
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

		extList.clear();
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

void DialogEvent::SetCropSizeList(const boost::filesystem::path & input_path)
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

void DialogEvent::ProcessWaifu2x()
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

			if (!outDir.empty() && !boost::filesystem::exists(outDir))
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

	if (input_str_multi.size() == 0)
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
	if (ret != Waifu2x::eWaifu2xError_OK)
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
			if (isOutputNoOverwrite && boost::filesystem::exists(p.second)) // 上書き禁止ならメッセージ表示して無視
			{
				SendMessage(dh, WM_ON_WAIFU2X_NO_OVERWRITE, (WPARAM)p.first.c_str(), (LPARAM)p.second.c_str());

				num++;
				ProgessFunc(maxFile, num);

				count++;

				continue;
			}

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
			else if (count >= 2)
				processeNum++;

			if (count == 1) // 最初の一回目は二回目以降より遅くなるはずなので残り時間の計算には使わない
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

void DialogEvent::ReplaceAddString() // ファイル名の自動設定部分を書き換える
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

void DialogEvent::AddLogMessage(const TCHAR * msg)
{
	if (logMessage.length() == 0)
		logMessage += msg;
	else
		logMessage += tstring(TEXT("\r\n")) + msg;

	SetWindowText(GetDlgItem(dh, IDC_EDIT_LOG), logMessage.c_str());
}

void DialogEvent::Waifu2xTime()
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

void DialogEvent::SaveIni(const bool isSyncMember)
{
	if (isSyncMember)
		SyncMember(true);

	if (isNotSaveParam)
		return;

	const boost::filesystem::path SettingFilePath(exeDir / SettingFileName);

	tstring tScaleRatio;
	tstring tScaleWidth;
	tstring tScaleHeight;
	tstring tmode;
	tstring tScaleMode;
	tstring tprcess;

	if (scale_ratio > 0.0)
		tScaleRatio = to_tstring(scale_ratio);
	else
		tScaleRatio = TEXT("");

	if (scale_width > 0)
		tScaleWidth = to_tstring(scale_width);
	else
		tScaleWidth = TEXT("");

	if (scale_height > 0)
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

	if (output_quality)
		WritePrivateProfileString(TEXT("Setting"), TEXT("LastOutputQuality"), boost::lexical_cast<tstring>(*output_quality).c_str(), getTString(SettingFilePath).c_str());
	else
		WritePrivateProfileString(TEXT("Setting"), TEXT("LastOutputQuality"), TEXT(""), getTString(SettingFilePath).c_str());

	WritePrivateProfileString(TEXT("Setting"), TEXT("LastOutputDepth"), boost::lexical_cast<tstring>(output_depth).c_str(), getTString(SettingFilePath).c_str());

	WritePrivateProfileString(TEXT("Setting"), TEXT("LastLanguage"), LangName.c_str(), getTString(SettingFilePath).c_str());

	WritePrivateProfileString(TEXT("Setting"), TEXT("LastAutoMode"), tAutoMode.c_str(), getTString(SettingFilePath).c_str());

	WritePrivateProfileString(TEXT("Setting"), TEXT("LastIsArgStartAuto"), to_tstring(isArgStartAuto ? 1 : 0).c_str(), getTString(SettingFilePath).c_str());
	WritePrivateProfileString(TEXT("Setting"), TEXT("LastIsArgStartSuccessFinish"), to_tstring(isArgStartSuccessFinish ? 1 : 0).c_str(), getTString(SettingFilePath).c_str());
	WritePrivateProfileString(TEXT("Setting"), TEXT("LastIsOutputNoOverwrite"), to_tstring(isOutputNoOverwrite ? 1 : 0).c_str(), getTString(SettingFilePath).c_str());

	WritePrivateProfileString(TEXT("Setting"), TEXT("LastInputDirFix"), tInputDirFix.c_str(), getTString(SettingFilePath).c_str());
	WritePrivateProfileString(TEXT("Setting"), TEXT("LastOutputDirFix"), tOutputDirFix.c_str(), getTString(SettingFilePath).c_str());
}

struct stFindParam
{
	const TCHAR *WindowName;
	HWND hWnd;
};

static BOOL CALLBACK EnumChildWindowsProc(HWND hWnd, LPARAM lParam)
{
	stFindParam *ptr = (stFindParam *)lParam;

	TCHAR buf[100];

	if (GetWindowTextLength(hWnd) > _countof(buf) - 1)
		return TRUE;

	GetWindowText(hWnd, buf, _countof(buf));
	buf[_countof(buf) - 1] = TEXT('\0');

	if (_tcscmp(ptr->WindowName, buf) == 0)
	{
		ptr->hWnd = hWnd;
		return FALSE;
	}

	return TRUE;
}

// 入力パスを選択する
UINT_PTR DialogEvent::OFNHookProcIn(HWND hdlg, UINT uiMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uiMsg)
	{
	case WM_INITDIALOG:
	{
		// ダイアログを中央に表示

		HWND hParent = GetParent(hdlg);

		HWND   hwndScreen;
		RECT   rectScreen = {0};
		hwndScreen = GetDesktopWindow();
		GetWindowRect(hwndScreen, &rectScreen);

		RECT rDialog = {0};
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

			stFindParam param;
			param.WindowName = TEXT("FolderView");
			param.hWnd = NULL;

			EnumChildWindows(hParent, EnumChildWindowsProc, (LPARAM)&param);

			if (param.hWnd)
			{
				std::vector< tstring >	results;
				int	index = -1;
				while (-1 != (index = ListView_GetNextItem(param.hWnd, index, LVNI_ALL | LVNI_SELECTED)))
				{
					std::vector< _TCHAR >	result(AR_PATH_MAX, TEXT('\0'));
					ListView_GetItemText(param.hWnd, index, 0, &result[0], result.size());
					results.push_back(result.data());
				}

				if (results.size() > 1)
				{
					TCHAR str[10000] = TEXT("");

					for (const auto &p : results)
					{
						_tcscat_s(str, TEXT("\""));
						_tcscat_s(str, p.c_str());
						_tcscat_s(str, TEXT("\" "));
					}

					CommDlg_OpenSave_SetControlText(hParent, edt1, str);
				}
				else if(results.size() == 1)
					CommDlg_OpenSave_SetControlText(hParent, edt1, results[0].c_str());
				else
					CommDlg_OpenSave_SetControlText(hParent, edt1, TEXT(""));
			}
		}
		break;
		}
	}
	break;
	}

	return 0L;
}

// 出力パスを選択する
UINT_PTR DialogEvent::OFNHookProcOut(HWND hdlg, UINT uiMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uiMsg)
	{
	case WM_INITDIALOG:
	{
		// ダイアログを中央に表示

		HWND hParent = GetParent(hdlg);

		HWND   hwndScreen;
		RECT   rectScreen = {0};
		hwndScreen = GetDesktopWindow();
		GetWindowRect(hwndScreen, &rectScreen);

		RECT rDialog = {0};
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
		// フォルダのみを選択できるようにする

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
				if (boost::filesystem::is_empty(szPath) || boost::filesystem::is_directory(szPath))
				{
					const auto filename = getTString(p.filename());

					CommDlg_OpenSave_SetControlText(hParent, edt1, filename.c_str());
				}
				else
					CommDlg_OpenSave_SetControlText(hParent, edt1, TEXT(""));
			}
		}
		break;
		}
	}
	break;
	}

	return 0L;
}

DialogEvent::DialogEvent() : dh(nullptr), mode("noise_scale"), noise_level(1), scale_ratio(2.0), scale_width(0), scale_height(0), model_dir(TEXT("models/anime_style_art_rgb")),
process("gpu"), outputExt(TEXT(".png")), inputFileExt(TEXT("png:jpg:jpeg:tif:tiff:bmp:tga")),
use_tta(false), output_depth(8), crop_size(128), batch_size(1), isLastError(false), scaleType(eScaleTypeEnd),
TimeLeftThread(-1), TimeLeftGetTimeThread(0), isCommandLineStart(false), tAutoMode(TEXT("none")),
isArgStartAuto(true), isArgStartSuccessFinish(true), isOutputNoOverwrite(false), isNotSaveParam(false)
{}

void DialogEvent::Exec(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
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

void DialogEvent::WaitThreadExit(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
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

			if (isCommandLineStart && isArgStartSuccessFinish) // コマンドライン引数を渡されて起動して、変換に成功したら終了する(フラグ設定時のみ)
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

void DialogEvent::Timer(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
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

void DialogEvent::OnDialogEnd(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	SaveIni();

	if (!processThread.joinable())
		PostQuitMessage(0);
	else
		MessageBeep(MB_ICONEXCLAMATION);
}

void DialogEvent::OnFaildCreateDir(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	const boost::filesystem::path *p = (const boost::filesystem::path *)wParam;

	TCHAR msg[1024 * 2];
	_stprintf(msg, langStringList.GetString(L"MessageCreateOutDirError").c_str(), getTString(*p).c_str());

	MessageBox(dh, msg, langStringList.GetString(L"MessageTitleError").c_str(), MB_OK | MB_ICONERROR);

	isLastError = true;
}

void DialogEvent::OnWaifu2xError(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
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

void DialogEvent::OnWaifu2xNoOverwrite(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	const TCHAR *input = (const TCHAR *)wParam;
	const TCHAR *output = (const TCHAR *)lParam;

	TCHAR msg[1024] = TEXT("");
	_stprintf(msg, langStringList.GetString(L"MessageNoOverwrite").c_str(), output);

	AddLogMessage(msg);
}

void DialogEvent::SetWindowTextLang()
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
	SET_WINDOW_TEXT(IDC_RADIONOISE_LEVEL3);
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
	SET_WINDOW_TEXT(IDC_STATIC_CROP_SIZE);
	SET_WINDOW_TEXT(IDC_BUTTON_CHECK_CUDNN);
	SET_WINDOW_TEXT(IDC_BUTTON_CANCEL);
	SET_WINDOW_TEXT(IDC_BUTTON_EXEC);
	SET_WINDOW_TEXT(IDC_STATIC_LANG_UI);
	SET_WINDOW_TEXT(IDC_RADIO_AUTO_START_NONE);
	SET_WINDOW_TEXT(IDC_RADIO_AUTO_START_ONE);
	SET_WINDOW_TEXT(IDC_RADIO_AUTO_START_MULTI);
	SET_WINDOW_TEXT(IDC_BUTTON_OUTPUT_REF);
	SET_WINDOW_TEXT(IDC_BUTTON_APP_SETTING);
	SET_WINDOW_TEXT(IDC_BUTTON_CLEAR_OUTPUT_DIR);

#undef SET_WINDOW_TEXT
}

void DialogEvent::SetDepthAndQuality(const bool SetDefaultQuality)
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
		if (SetDefaultQuality)
			SetWindowText(hedit, boost::lexical_cast<tstring>(*elm.imageQualityDefault).c_str());

		const auto wstr = langStringList.GetString(L"IDC_STATIC_OUTPUT_QUALITY");

		const auto addstr = std::wstring(L" (") + boost::lexical_cast<std::wstring>(*elm.imageQualityStart)
			+ L"～" + boost::lexical_cast<std::wstring>(*elm.imageQualityEnd) + L")";
		SetWindowTextW(GetDlgItem(dh, IDC_STATIC_OUTPUT_QUALITY), (wstr + addstr).c_str());
	}
}

void DialogEvent::Create(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
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

		isArgStartAuto = GetPrivateProfileInt(TEXT("Setting"), TEXT("LastIsArgStartAuto"), 1, getTString(SettingFilePath).c_str()) != 0;
		isArgStartSuccessFinish = GetPrivateProfileInt(TEXT("Setting"), TEXT("LastIsArgStartSuccessFinish"), 1, getTString(SettingFilePath).c_str()) != 0;
		isOutputNoOverwrite = GetPrivateProfileInt(TEXT("Setting"), TEXT("LastIsOutputNoOverwrite"), 0, getTString(SettingFilePath).c_str()) != 0;

		GetPrivateProfileString(TEXT("Setting"), TEXT("LastInputDirFix"), TEXT(""), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
		tmp[_countof(tmp) - 1] = TEXT('\0');
		tInputDirFix = tmp;

		GetPrivateProfileString(TEXT("Setting"), TEXT("LastOutputDirFix"), TEXT(""), tmp, _countof(tmp), getTString(SettingFilePath).c_str());
		tmp[_countof(tmp) - 1] = TEXT('\0');
		tOutputDirFix = tmp;
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

	if (!(1 <= noise_level && noise_level <= 3))
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

	if (tScaleMode == TEXT("Ratio"))
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
		SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL3), BM_SETCHECK, BST_UNCHECKED, 0);
	}
	else if (noise_level == 2)
	{
		SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL1), BM_SETCHECK, BST_UNCHECKED, 0);
		SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL2), BM_SETCHECK, BST_CHECKED, 0);
		SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL3), BM_SETCHECK, BST_UNCHECKED, 0);
	}
	else
	{
		SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL1), BM_SETCHECK, BST_UNCHECKED, 0);
		SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL2), BM_SETCHECK, BST_UNCHECKED, 0);
		SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL3), BM_SETCHECK, BST_CHECKED, 0);
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

	if (tOutputDirFix.length() > 0 && boost::filesystem::exists(tOutputDirFix))
	{
		output_dir = tOutputDirFix;
		SetWindowText(GetDlgItem(hWnd, IDC_EDIT_OUTPUT), output_dir.c_str());
	}

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

	if (output_quality)
		SetWindowText(GetDlgItem(hWnd, IDC_EDIT_OUT_QUALITY), boost::lexical_cast<tstring>(*output_quality).c_str());

	SetWindowText(GetDlgItem(hWnd, IDC_COMBO_OUTPUT_DEPTH), boost::lexical_cast<tstring>(output_depth).c_str());

	SetDepthAndQuality(false);

	int nArgs = 0;
	LPTSTR *lplpszArgs;
	lplpszArgs = CommandLineToArgvW(GetCommandLine(), &nArgs);
	if (lplpszArgs)
	{
		input_str_multi.clear();

		if (nArgs > 1)
		{
			// definition of command line arguments
			TCLAP::CmdLine cmd(L"waifu2x reimplementation using Caffe", L' ', L"1.0.0");

			// GUIでは-iを付けない
			TCLAP::UnlabeledMultiArg<std::wstring> cmdInputFile(L"input_file_paths", L"input file paths", false,
				L"string", cmd);

			// GUIでは出力先フォルダのみの指定
			TCLAP::ValueArg<std::wstring> cmdOutputDir(L"o", L"output_folder",
				L"path to output image folder", false,
				L"", L"string", cmd);

			TCLAP::ValueArg<std::wstring> cmdInputFileExt(L"l", L"input_extention_list",
				L"extention to input image file when input_path is folder", false, L"png:jpg:jpeg:tif:tiff:bmp:tga",
				L"string", cmd);

			TCLAP::ValueArg<std::wstring> cmdOutputFileExt(L"e", L"output_extention",
				L"extention to output image file when output_path is (auto) or input_path is folder", false,
				L"png", L"string", cmd);

			std::vector<std::wstring> cmdModeConstraintV;
			cmdModeConstraintV.push_back(L"noise");
			cmdModeConstraintV.push_back(L"scale");
			cmdModeConstraintV.push_back(L"noise_scale");
			cmdModeConstraintV.push_back(L"auto_scale");
			TCLAP::ValuesConstraint<std::wstring> cmdModeConstraint(cmdModeConstraintV);
			TCLAP::ValueArg<std::wstring> cmdMode(L"m", L"mode", L"image processing mode",
				false, L"noise_scale", &cmdModeConstraint, cmd);

			std::vector<int> cmdNRLConstraintV;
			cmdNRLConstraintV.push_back(1);
			cmdNRLConstraintV.push_back(2);
			cmdNRLConstraintV.push_back(3);
			TCLAP::ValuesConstraint<int> cmdNRLConstraint(cmdNRLConstraintV);
			TCLAP::ValueArg<int> cmdNRLevel(L"n", L"noise_level", L"noise reduction level",
				false, 1, &cmdNRLConstraint, cmd);

			TCLAP::ValueArg<double> cmdScaleRatio(L"s", L"scale_ratio",
				L"custom scale ratio", false, 2.0, L"double", cmd);

			TCLAP::ValueArg<int> cmdScaleWidth(L"w", L"scale_width",
				L"custom scale width", false, 0, L"double", cmd);

			TCLAP::ValueArg<int> cmdScaleHeight(L"h", L"scale_height",
				L"custom scale height", false, 0, L"double", cmd);

			std::vector<std::wstring> cmdProcessConstraintV;
			cmdProcessConstraintV.push_back(L"cpu");
			cmdProcessConstraintV.push_back(L"gpu");
			TCLAP::ValuesConstraint<std::wstring> cmdProcessConstraint(cmdProcessConstraintV);
			TCLAP::ValueArg<std::wstring> cmdProcess(L"p", L"process", L"process mode",
				false, L"gpu", &cmdProcessConstraint, cmd);

			TCLAP::ValueArg<int> cmdOutputQuality(L"q", L"output_quality",
				L"output image quality", false,
				-1, L"int", cmd);

			TCLAP::ValueArg<int> cmdOutputDepth(L"d", L"output_depth",
				L"output image chaneel depth bit", false,
				8, L"int", cmd);

			TCLAP::ValueArg<int> cmdCropSizeFile(L"c", L"crop_size",
				L"input image split size", false,
				128, L"int", cmd);

			TCLAP::ValueArg<int> cmdBatchSizeFile(L"b", L"batch_size",
				L"input batch size", false,
				1, L"int", cmd);

			std::vector<int> cmdBoolConstraintV;
			cmdBoolConstraintV.push_back(0);
			cmdBoolConstraintV.push_back(1);

			TCLAP::ValuesConstraint<int> cmdTTAConstraint(cmdBoolConstraintV);
			TCLAP::ValueArg<int> cmdTTA(L"t", L"tta", L"8x slower and slightly high quality",
				false, 0, &cmdTTAConstraint, cmd);

			// GUI独自
			TCLAP::ValuesConstraint<int> cmdAutoStartConstraint(cmdBoolConstraintV);
			TCLAP::ValueArg<int> cmdAutoStart(L"", L"auto_start", L"to run automatically at startup",
				false, 0, &cmdAutoStartConstraint, cmd);

			TCLAP::ValuesConstraint<int> cmdAutoExitConstraint(cmdBoolConstraintV);
			TCLAP::ValueArg<int> cmdAutoExit(L"", L"auto_exit", L"exit when the run was succeeded",
				false, 0, &cmdAutoExitConstraint, cmd);

			TCLAP::ValuesConstraint<int> cmdNoOverwriteConstraint(cmdBoolConstraintV);
			TCLAP::ValueArg<int> cmdNoOverwrite(L"", L"no_overwrite", L"don't overwrite output file",
				false, 0, &cmdNoOverwriteConstraint, cmd);

			std::vector<std::wstring> cmdModelTypeConstraintV;
			cmdModelTypeConstraintV.push_back(L"anime_style_art_rgb");
			cmdModelTypeConstraintV.push_back(L"photo");
			cmdModelTypeConstraintV.push_back(L"anime_style_art_y");
			TCLAP::ValuesConstraint<std::wstring> cmdModelTypeConstraint(cmdModelTypeConstraintV);
			TCLAP::ValueArg<std::wstring> cmdModelType(L"y", L"model_type", L"model type",
				false, L"anime_style_art_rgb", &cmdModelTypeConstraint, cmd);

			// definition of command line argument : end

			TCLAP::Arg::enableIgnoreMismatched();

			// parse command line arguments
			try
			{
				cmd.parse(nArgs, lplpszArgs);

				bool isSetParam = false;

				if (cmdOutputDir.isSet())
				{
					OnSetOutputFilePath(cmdOutputDir.getValue().c_str());

					isSetParam = true;
				}

				if (cmdInputFileExt.isSet())
				{
					const auto inputFileExt = cmdInputFileExt.getValue();
					SetWindowText(GetDlgItem(dh, IDC_EDIT_INPUT_EXT_LIST), inputFileExt.c_str());

					isSetParam = true;
				}

				if (cmdOutputFileExt.isSet())
				{
					auto OutputExt = cmdInputFileExt.getValue();
					if (OutputExt.length() > 0 && OutputExt[0] != L'.')
						OutputExt.insert(0, L".");
					SetWindowText(GetDlgItem(dh, IDC_COMBO_OUT_EXT), OutputExt.c_str());

					isSetParam = true;
				}

				if (cmdMode.isSet())
				{
					tmode = cmdMode.getValue();

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

					isSetParam = true;
				}

				if (cmdNRLevel.isSet())
				{
					const auto noise_level = cmdNRLevel.getValue();

					if (noise_level == 1)
					{
						SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL1), BM_SETCHECK, BST_CHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL2), BM_SETCHECK, BST_UNCHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL3), BM_SETCHECK, BST_UNCHECKED, 0);
					}
					else if (noise_level == 2)
					{
						SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL1), BM_SETCHECK, BST_UNCHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL2), BM_SETCHECK, BST_CHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL3), BM_SETCHECK, BST_UNCHECKED, 0);
					}
					else
					{
						SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL1), BM_SETCHECK, BST_UNCHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL2), BM_SETCHECK, BST_UNCHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL3), BM_SETCHECK, BST_CHECKED, 0);
					}

					isSetParam = true;
				}

				if (cmdScaleWidth.isSet())
				{
					SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_RATIO), BM_SETCHECK, BST_UNCHECKED, 0);
					SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_WIDTH), BM_SETCHECK, BST_CHECKED, 0);
					SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_HEIGHT), BM_SETCHECK, BST_UNCHECKED, 0);

					EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), FALSE);
					EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), TRUE);
					EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), FALSE);

					SetWindowText(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), to_tstring(cmdScaleWidth.getValue()).c_str());

					isSetParam = true;
				}
				else if (cmdScaleHeight.isSet())
				{
					SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_RATIO), BM_SETCHECK, BST_UNCHECKED, 0);
					SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_WIDTH), BM_SETCHECK, BST_UNCHECKED, 0);
					SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_HEIGHT), BM_SETCHECK, BST_CHECKED, 0);

					EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), FALSE);
					EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), FALSE);
					EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), TRUE);

					SetWindowText(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), to_tstring(cmdScaleHeight.getValue()).c_str());

					isSetParam = true;
				}
				else if (cmdScaleRatio.isSet())
				{
					SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_RATIO), BM_SETCHECK, BST_CHECKED, 0);
					SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_WIDTH), BM_SETCHECK, BST_UNCHECKED, 0);
					SendMessage(GetDlgItem(hWnd, IDC_RADIO_SCALE_HEIGHT), BM_SETCHECK, BST_UNCHECKED, 0);

					EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), TRUE);
					EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), FALSE);
					EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), FALSE);

					SetWindowText(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), to_tstring(cmdScaleRatio.getValue()).c_str());

					isSetParam = true;
				}

				if (cmdProcess.isSet())
				{
					if (cmdProcess.getValue() == L"gpu")
					{
						SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_GPU), BM_SETCHECK, BST_CHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_CPU), BM_SETCHECK, BST_UNCHECKED, 0);
					}
					else
					{
						SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_GPU), BM_SETCHECK, BST_UNCHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_CPU), BM_SETCHECK, BST_CHECKED, 0);
					}

					isSetParam = true;
				}

				if (cmdOutputQuality.isSet())
				{
					SetWindowText(GetDlgItem(dh, IDC_EDIT_OUT_QUALITY), to_tstring(cmdOutputQuality.getValue()).c_str());

					isSetParam = true;
				}

				if (cmdOutputDepth.isSet())
				{
					SetWindowText(GetDlgItem(dh, IDC_COMBO_OUTPUT_DEPTH), to_tstring(cmdOutputDepth.getValue()).c_str());

					isSetParam = true;
				}

				if (cmdCropSizeFile.isSet())
				{
					SetWindowText(GetDlgItem(dh, IDC_COMBO_CROP_SIZE), to_tstring(cmdCropSizeFile.getValue()).c_str());

					isSetParam = true;
				}

				if (cmdBatchSizeFile.isSet())
				{
					batch_size = cmdBatchSizeFile.getValue();

					isSetParam = true;
				}

				if (cmdTTA.isSet())
				{
					SendMessage(GetDlgItem(dh, IDC_CHECK_TTA), BM_SETCHECK, cmdTTA.getValue() ? BST_CHECKED : BST_UNCHECKED, 0);

					isSetParam = true;
				}

				if (cmdAutoStart.isSet())
				{
					isArgStartAuto = cmdAutoStart.getValue() != 0;

					isSetParam = true;
				}

				if (cmdAutoExit.isSet())
				{
					isArgStartSuccessFinish = cmdAutoExit.getValue() != 0;

					isSetParam = true;
				}

				if (cmdModelType.isSet())
				{
					if (cmdModelType.getValue() == L"anime_style_art_rgb")
					{
						SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_RGB), BM_SETCHECK, BST_CHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_PHOTO), BM_SETCHECK, BST_UNCHECKED, 0);
						SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODEL_Y), BM_SETCHECK, BST_UNCHECKED, 0);
					}
					else if (cmdModelType.getValue() == L"photo")
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

					isSetParam = true;
				}

				if (cmdNoOverwrite.isSet())
				{
					isOutputNoOverwrite = cmdNoOverwrite.getValue() != 0;

					isSetParam = true;
				}

				if (isSetParam)
					isNotSaveParam = true;

				const auto &vi = cmdInputFile.getValue();
				if (vi.size() > 1)
				{
					for (size_t i = 1; i < vi.size(); i++)
					{
						input_str_multi.push_back(vi[i]);
					}

					OnSetInputFilePath();
				}
				else if (vi.size() == 1)
				{
					OnSetInputFilePath(vi[0].c_str());
				}

				if (isArgStartAuto) // 引数指定されたら自動で実行(フラグ設定時のみ)
				{
					isCommandLineStart = true;
					::PostMessage(GetDlgItem(dh, IDC_BUTTON_EXEC), BM_CLICK, 0, 0);
				}
			}
			catch (std::exception &e)
			{
			}
		}

		LocalFree(lplpszArgs);
	}
}

void DialogEvent::Cancel(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	cancelFlag = true;
	EnableWindow(GetDlgItem(dh, IDC_BUTTON_CANCEL), FALSE);
}

void DialogEvent::UpdateAddString(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	ReplaceAddString();
}

void DialogEvent::OnModeChange(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	bool isNoise = false;
	bool isScale = false;

	if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODE_NOISE), BM_GETCHECK, 0, 0))
	{
		isNoise = true;
		isScale = false;
	}
	else if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODE_SCALE), BM_GETCHECK, 0, 0))
	{
		isNoise = false;
		isScale = true;
	}
	else if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODE_NOISE_SCALE), BM_GETCHECK, 0, 0))
	{
		isNoise = true;
		isScale = true;
	}
	else
	{
		isNoise = true;
		isScale = true;
	}

	if (isNoise)
	{
		EnableWindow(GetDlgItem(dh, IDC_RADIONOISE_LEVEL1), TRUE);
		EnableWindow(GetDlgItem(dh, IDC_RADIONOISE_LEVEL2), TRUE);
		EnableWindow(GetDlgItem(dh, IDC_RADIONOISE_LEVEL3), TRUE);
	}
	else
	{
		EnableWindow(GetDlgItem(dh, IDC_RADIONOISE_LEVEL1), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_RADIONOISE_LEVEL2), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_RADIONOISE_LEVEL3), FALSE);
	}

	if (isScale)
	{
		EnableWindow(GetDlgItem(dh, IDC_RADIO_SCALE_RATIO), TRUE);
		EnableWindow(GetDlgItem(dh, IDC_RADIO_SCALE_WIDTH), TRUE);
		EnableWindow(GetDlgItem(dh, IDC_RADIO_SCALE_HEIGHT), TRUE);

		ScaleRadio(NULL, NULL, NULL, NULL); // ここでReplaceAddString()やるからreturn
		return;
	}
	else
	{
		EnableWindow(GetDlgItem(dh, IDC_RADIO_SCALE_RATIO), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_RADIO_SCALE_WIDTH), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_RADIO_SCALE_HEIGHT), FALSE);

		EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_WIDTH), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_EDIT_SCALE_HEIGHT), FALSE);
	}

	ReplaceAddString();
}

void DialogEvent::ScaleRadio(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
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

void DialogEvent::CheckCUDNN(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
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
	case Waifu2x::eWaifu2xCudaError_OldDevice:
		MessageBox(dh, langStringList.GetString(L"MessageCudaOldDeviceError").c_str(), langStringList.GetString(L"MessageTitleResult").c_str(), MB_OK | MB_ICONERROR);
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

LRESULT DialogEvent::OnSetInputFilePath(const TCHAR * tPath)
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

	boost::filesystem::path outpath(output_dir);

	if (boost::filesystem::is_directory(path))
	{
		HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

		if (output_dir.length() == 0) // 出力パス未設定なら入力ファイルのフォルダ
			outpath = path.branch_path();

		const tstring addstr(AddName());
		autoSetAddName = AddName();

		const auto str = getTString(outpath / (path.stem().wstring() + addstr));
		SetWindowText(ho, str.c_str());

		SetWindowText(hWnd, tPath);
	}
	else
	{
		HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

		if (output_dir.length() == 0) // 出力パス未設定なら入力フォルダと同じフォルダ
			outpath = path.branch_path();

		tstring outputFileName = getTString(path.filename());

		const auto tailDot = outputFileName.find_last_of(TEXT('.'));
		if (tailDot != outputFileName.npos)
			outputFileName.erase(tailDot, outputFileName.length());

		const tstring addstr(AddName());
		autoSetAddName = addstr;

		outpath = outpath / (outputFileName + addstr + outputExt);

		outputFileName = getTString(outpath);

		SetWindowText(ho, outputFileName.c_str());

		SetWindowText(hWnd, tPath);
	}

	SetCropSizeList(path);

	return 0L;
}

LRESULT DialogEvent::OnSetInputFilePath()
{
	HWND hWnd = GetDlgItem(dh, IDC_EDIT_INPUT);

	SyncMember(true, true);

	if (input_str_multi.size() > 0)
	{
		SetWindowText(hWnd, MultiFileStr);

		HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

		const tstring addstr(AddName());
		autoSetAddName = AddName();

		boost::filesystem::path outpath(output_dir);

		if (output_dir.length() == 0) // 出力パス未設定なら入力ファイルと同じフォルダ
		{
			outpath = input_str_multi[0];
			outpath = outpath.branch_path();
		}

		boost::filesystem::path baseDir(input_str_multi[0]);

		tstring filename;
		if (boost::filesystem::is_directory(baseDir))
			filename = baseDir.filename().wstring();
		else
			filename = baseDir.stem().wstring();

		const auto str = getTString(outpath / (filename + TEXT(" multi") + addstr));
		SetWindowText(ho, str.c_str());
	}

	SetCropSizeList(TEXT(""));

	return 0L;
}

LRESULT DialogEvent::OnSetOutputFilePath(const TCHAR * tPath)
{
	HWND hWnd = GetDlgItem(dh, IDC_EDIT_OUTPUT);

	SyncMember(true, true);

	if (input_str.length() > 0 || input_str_multi.size() > 0)
	{
		boost::filesystem::path path(input_str);
		boost::filesystem::path outpath(tPath);

		if (input_str_multi.size() > 0)
		{
			path = input_str_multi[0];
		}

		if (boost::filesystem::is_directory(path))
		{
			HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

			const tstring addstr(AddName());
			autoSetAddName = AddName();

			output_str = getTString(outpath / (path.stem().wstring() + addstr));
			SetWindowText(ho, output_str.c_str());
		}
		else
		{
			HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

			tstring outputFileName = getTString(path.filename());

			const auto tailDot = outputFileName.find_last_of(TEXT('.'));
			if (tailDot != outputFileName.npos)
				outputFileName.erase(tailDot, outputFileName.length());

			const tstring addstr(AddName());
			autoSetAddName = addstr;

			outpath = outpath / (outputFileName + addstr + outputExt);

			output_str = getTString(outpath);

			SetWindowText(ho, output_str.c_str());
		}
	}
	else
	{
		SetWindowText(hWnd, tPath);
		output_str = tPath;
	}

	output_dir = tPath;

	return 0L;
}

// ここで渡されるhWndはIDC_EDITのHWND(コントロールのイベントだから)

LRESULT DialogEvent::DropInput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData)
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

		if (tAutoMode == TEXT("one") ||
			(tAutoMode == TEXT("multi") && (input_str_multi.size() > 0 || boost::filesystem::is_directory(szTmp))))
		{
			::PostMessage(GetDlgItem(dh, IDC_BUTTON_EXEC), BM_CLICK, 0, 0);
		}
	}

	return 0L;
}

// ここで渡されるhWndはIDC_EDITのHWND(コントロールのイベントだから)

LRESULT DialogEvent::DropOutput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData)
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

LRESULT DialogEvent::TextInput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData)
{
	const auto ret = CallWindowProc(OrgSubWnd, hWnd, WM_CHAR, wParam, lParam);
	ReplaceAddString();
	return ret;
}

void DialogEvent::InputRef(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	SyncMember(false);

	OPENFILENAME ofn = {0};
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

	if (tInputDirFix.length() > 0 && boost::filesystem::exists(tInputDirFix))
		ofn.lpstrInitialDir = tInputDirFix.c_str();
	else
		ofn.lpstrInitialDir = szPath;

	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = dh;
	ofn.lpstrFile = szFile.data();
	ofn.nMaxFile = szFile.size();
	ofn.lpstrFilter = szFilter;
	ofn.nFilterIndex = 1;
	ofn.lpstrTitle = langStringList.GetString(L"MessageTitleInputDialog").c_str();
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

			if (firstLen + 2 >= szFile.size() || ptr[firstLen + 1] == '\0')
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

			if (tAutoMode == TEXT("one") ||
				(tAutoMode == TEXT("multi") && (input_str_multi.size() > 0 || boost::filesystem::is_directory(szFile.data()))))
			{
				::PostMessage(GetDlgItem(dh, IDC_BUTTON_EXEC), BM_CLICK, 0, 0);
			}
		}
	}
}

void DialogEvent::OutputRef(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	SyncMember(false);

	OPENFILENAME ofn = {0};
	TCHAR szPath[AR_PATH_MAX] = TEXT("");

	static std::vector<TCHAR> szFile(AR_PATH_MAX * 100);
	memset(szFile.data(), 0, szFile.size() * sizeof(TCHAR));

	GetCurrentDirectory(_countof(szPath), szPath);
	szPath[_countof(szPath) - 1] = TEXT('\0');


	TCHAR szFilter[AR_PATH_MAX] = TEXT("");
	TCHAR *tfp = szFilter;

	const tstring allFilesTitle(langStringList.GetString(L"MessageAllFileFolder").c_str());
	memcpy(tfp, allFilesTitle.c_str(), allFilesTitle.length() * sizeof(TCHAR));
	tfp += allFilesTitle.length();
	*tfp = TEXT('\0');
	tfp++;

	const tstring allFilesExt(TEXT("*.*"));
	memcpy(tfp, allFilesExt.c_str(), allFilesExt.length() * sizeof(TCHAR));
	tfp += allFilesExt.length();

	if (tOutputDirFix.length() > 0 && boost::filesystem::exists(tOutputDirFix))
		ofn.lpstrInitialDir = tOutputDirFix.c_str();
	else
		ofn.lpstrInitialDir = szPath;

	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = dh;
	ofn.lpstrFile = szFile.data();
	ofn.nMaxFile = szFile.size();
	ofn.lpstrFilter = szFilter;
	ofn.nFilterIndex = 1;
	ofn.lpstrTitle = langStringList.GetString(L"MessageTitleInputDialog").c_str();
	ofn.lpstrCustomFilter = NULL;
	ofn.nMaxCustFilter = 0;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.nFileOffset = 0;
	ofn.nFileExtension = 0;
	ofn.lpstrDefExt = NULL;
	ofn.lCustData = 0;
	ofn.lpfnHook = OFNHookProcOut;
	ofn.lpTemplateName = 0;
	ofn.Flags = OFN_HIDEREADONLY | OFN_NOVALIDATE | OFN_ENABLESIZING | OFN_OVERWRITEPROMPT | OFN_EXPLORER | OFN_ENABLEHOOK;
	if (GetOpenFileName(&ofn))
	{
		szFile[szFile.size() - 1] = TEXT('\0');

		const TCHAR * ptr = szFile.data();

		const auto firstLen = _tcslen(ptr);
		if (firstLen > 0)
		{
			output_str.clear();

			if (firstLen > 0)
			{
				OnSetOutputFilePath(ptr);
			}
		}
	}
}

void DialogEvent::LangChange(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
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

void DialogEvent::OutExtChange(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	if (HIWORD(wParam) != CBN_SELCHANGE)
		return;

	SetDepthAndQuality();

	ReplaceAddString();
}

void DialogEvent::ClearOutputDir(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	output_dir.clear();

	SyncMember(true, true);

	if(input_str.length() > 0)
		OnSetInputFilePath(input_str.c_str());
	else
		OnSetInputFilePath();
}

void DialogEvent::AppSetting(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
{
	class AppSettingDialogEvent
	{
	private:
		HWND dh;
		LangStringList &langStringList;

	public:
		tstring tAutoMode;
		std::string process;
		bool isArgStartAuto;
		bool isArgStartSuccessFinish;
		bool isOutputNoOverwrite;
		tstring tInputDirFix;
		tstring tOutputDirFix;

	private:
		void AppSettingDialogEvent::SetWindowTextLang()
		{
			SetWindowTextW(dh, langStringList.GetString(L"IDC_BUTTON_APP_SETTING").c_str());
			SetWindowTextW(GetDlgItem(dh, IDOK), langStringList.GetString(L"OK").c_str());
			SetWindowTextW(GetDlgItem(dh, IDCANCEL), langStringList.GetString(L"Cancel").c_str());

#define SET_WINDOW_TEXT(id) SetWindowTextW(GetDlgItem(dh, id), langStringList.GetString(L#id).c_str());

			SET_WINDOW_TEXT(IDC_STATIC_AUTO_START);
			SET_WINDOW_TEXT(IDC_RADIO_AUTO_START_NONE);
			SET_WINDOW_TEXT(IDC_RADIO_AUTO_START_ONE);
			SET_WINDOW_TEXT(IDC_RADIO_AUTO_START_MULTI);

			SET_WINDOW_TEXT(IDC_STATIC_PROCESSOR);
			SET_WINDOW_TEXT(IDC_RADIO_MODE_GPU);
			SET_WINDOW_TEXT(IDC_RADIO_MODE_CPU);

			SET_WINDOW_TEXT(IDC_STATIC_ARG_START);
			SET_WINDOW_TEXT(IDC_CHECK_ARG_START_AUTO);
			SET_WINDOW_TEXT(IDC_CHECK_ARG_START_SUCCESS_FINISH);

			SET_WINDOW_TEXT(IDC_STATIC_INPUT_DIR_FIX);
			SET_WINDOW_TEXT(IDC_STATIC_OUTPUT_DIR_FIX);

			SET_WINDOW_TEXT(IDC_CHECK_OUTPUT_NO_OVERWIRITE);

#undef SET_WINDOW_TEXT
		}

	public:
		AppSettingDialogEvent(LangStringList &LangStringList) : dh(NULL), langStringList(LangStringList)
		{}

		void AppSettingDialogEvent::Create(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
		{
			dh = hWnd;

			SetWindowTextLang();

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

			if (isArgStartAuto)
				SendMessage(GetDlgItem(hWnd, IDC_CHECK_ARG_START_AUTO), BM_SETCHECK, BST_CHECKED, 0);

			if (isArgStartSuccessFinish)
				SendMessage(GetDlgItem(hWnd, IDC_CHECK_ARG_START_SUCCESS_FINISH), BM_SETCHECK, BST_CHECKED, 0);

			if (isOutputNoOverwrite)
				SendMessage(GetDlgItem(hWnd, IDC_CHECK_OUTPUT_NO_OVERWIRITE), BM_SETCHECK, BST_CHECKED, 0);

			SetWindowText(GetDlgItem(hWnd, IDC_EDIT_INPUT_DIR_FIX), tInputDirFix.c_str());
			SetWindowText(GetDlgItem(hWnd, IDC_EDIT_OUTPUT_DIR_FIX), tOutputDirFix.c_str());
		}

		void Close(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
		{
			sync();
		}

		void sync()
		{
			if (SendMessage(GetDlgItem(dh, IDC_RADIO_AUTO_START_ONE), BM_GETCHECK, 0, 0))
				tAutoMode = TEXT("one");
			else if (SendMessage(GetDlgItem(dh, IDC_RADIO_AUTO_START_MULTI), BM_GETCHECK, 0, 0))
				tAutoMode = TEXT("multi");
			else
				tAutoMode = TEXT("none");

			if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODE_CPU), BM_GETCHECK, 0, 0))
				process = "cpu";
			else
				process = "gpu";

			isArgStartAuto = SendMessage(GetDlgItem(dh, IDC_CHECK_ARG_START_AUTO), BM_GETCHECK, 0, 0) == BST_CHECKED;
			isArgStartSuccessFinish = SendMessage(GetDlgItem(dh, IDC_CHECK_ARG_START_SUCCESS_FINISH), BM_GETCHECK, 0, 0) == BST_CHECKED;
			isOutputNoOverwrite = SendMessage(GetDlgItem(dh, IDC_CHECK_OUTPUT_NO_OVERWIRITE), BM_GETCHECK, 0, 0) == BST_CHECKED;

			TCHAR buf[AR_PATH_MAX] = TEXT("");

			GetWindowText(GetDlgItem(dh, IDC_EDIT_INPUT_DIR_FIX), buf, _countof(buf));
			buf[_countof(buf) - 1] = TEXT('\0');
			tInputDirFix = buf;

			GetWindowText(GetDlgItem(dh, IDC_EDIT_OUTPUT_DIR_FIX), buf, _countof(buf));
			buf[_countof(buf) - 1] = TEXT('\0');
			tOutputDirFix = buf;
		}

		void AppSettingDialogEvent::OK(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
		{
			sync();
			EndDialog(dh, IDOK);
		}

		void AppSettingDialogEvent::CANCEL(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
		{
			sync();
			EndDialog(dh, IDCANCEL);
		}
	};

	AppSettingDialogEvent cAppSettingDialogEvent(langStringList);
	cAppSettingDialogEvent.tAutoMode = tAutoMode;
	cAppSettingDialogEvent.process = process;
	cAppSettingDialogEvent.isArgStartAuto = isArgStartAuto;
	cAppSettingDialogEvent.isArgStartSuccessFinish = isArgStartSuccessFinish;
	cAppSettingDialogEvent.isOutputNoOverwrite = isOutputNoOverwrite;
	cAppSettingDialogEvent.tInputDirFix = tInputDirFix;
	cAppSettingDialogEvent.tOutputDirFix = tOutputDirFix;

	CDialog cDialog;

	cDialog.SetCommandCallBack(SetClassFunc(AppSettingDialogEvent::OK, &cAppSettingDialogEvent), NULL, IDOK);
	cDialog.SetCommandCallBack(SetClassFunc(AppSettingDialogEvent::CANCEL, &cAppSettingDialogEvent), NULL, IDCANCEL);

	cDialog.SetEventCallBack(SetClassFunc(AppSettingDialogEvent::Create, &cAppSettingDialogEvent), NULL, WM_INITDIALOG);
	cDialog.SetEventCallBack(SetClassFunc(AppSettingDialogEvent::Close, &cAppSettingDialogEvent), NULL, WM_CLOSE);

	const int ret = cDialog.DoModal(GetModuleHandle(NULL), IDD_DIALOG_APP_SETTING, dh);

	if (ret == IDOK)
	{
		tAutoMode = cAppSettingDialogEvent.tAutoMode;
		process = cAppSettingDialogEvent.process;
		isArgStartAuto = cAppSettingDialogEvent.isArgStartAuto;
		isArgStartSuccessFinish = cAppSettingDialogEvent.isArgStartSuccessFinish;
		isOutputNoOverwrite = cAppSettingDialogEvent.isOutputNoOverwrite;
		tInputDirFix = cAppSettingDialogEvent.tInputDirFix;
		tOutputDirFix = cAppSettingDialogEvent.tOutputDirFix;

		if (tOutputDirFix.length() > 0 && boost::filesystem::exists(tOutputDirFix))
		{
			output_dir = tOutputDirFix;
		}
	}
}
