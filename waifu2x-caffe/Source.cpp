#include <stdio.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <functional>
#include <boost/tokenizer.hpp>
#include <boost/tokenizer.hpp>
#include <glog/logging.h>
#include <codecvt>
#include "../common/waifu2x.h"

#if defined(UNICODE) && defined(_WIN32)
#define WIN_UNICODE
#include <Windows.h>
#include <tclapw/CmdLine.h>
#include <io.h>
#include <fcntl.h>
using namespace TCLAPW;
typedef std::wstring tstring;
typedef wchar_t TCHAR;
#ifndef TEXT
#define TEXT(x) L##x
#endif
#define totlower towlower
#define to_tstring std::to_wstring
#define tprintf wprintf
#define CHAR_STR_FORMAT L"%S"

const tstring& path_to_tstring(const boost::filesystem::path &p)
{
	return p.wstring();
}
#else
#include <tclap/CmdLine.h>
using namespace TCLAP;
typedef std::string tstring;
typedef char TCHAR;
#ifndef TEXT
#define TEXT(x) x
#endif
#define totlower tolower
#define to_tstring std::to_string
#define tprintf printf
#define CHAR_STR_FORMAT "%s"

const tstring& path_to_tstring(const boost::filesystem::path &p)
{
	return p.string();
}
#endif


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

int main(int argc, char** argv)
{
#ifdef WIN_UNICODE
	//switch the console to UTF-16 mode
	_setmode(_fileno(stdout), _O_U16TEXT);
#endif

	Waifu2x::init_liblary(argc, argv);

	// Caffeのエラーでないログを保存しないようにする
	google::SetLogDestination(google::GLOG_INFO, "");
	google::SetLogDestination(google::GLOG_WARNING, "");

	// Caffeのエラーログを「error_log_～」に出力
	google::SetLogDestination(google::GLOG_ERROR, "error_log_");
	google::SetLogDestination(google::GLOG_FATAL, "error_log_");

	// definition of command line arguments
	CmdLine cmd(TEXT("waifu2x reimplementation using Caffe"), ' ', TEXT("1.0.0"));

	ValueArg<tstring> cmdInputFile(TEXT("i"), TEXT("input_path"),
		TEXT("path to input image file"), true, TEXT(""),
		TEXT("string"), cmd);

	ValueArg<tstring> cmdOutputFile(TEXT("o"), TEXT("output_path"),
		TEXT("path to output image file (when input_path is folder, output_path must be folder)"), false,
		TEXT("(auto)"), TEXT("string"), cmd);

	ValueArg<tstring> cmdInputFileExt(TEXT("l"), TEXT("input_extention_list"),
		TEXT("extention to input image file when input_path is folder"), false, TEXT("png:jpg:jpeg:tif:tiff:bmp:tga"),
		TEXT("string"), cmd);

	ValueArg<tstring> cmdOutputFileExt(TEXT("e"), TEXT("output_extention"),
		TEXT("extention to output image file when output_path is (auto) or input_path is folder"), false,
		TEXT("png"), TEXT("string"), cmd);

	std::vector<tstring> cmdModeConstraintV;
	cmdModeConstraintV.push_back(TEXT("noise"));
	cmdModeConstraintV.push_back(TEXT("scale"));
	cmdModeConstraintV.push_back(TEXT("noise_scale"));
	cmdModeConstraintV.push_back(TEXT("auto_scale"));
	ValuesConstraint<tstring> cmdModeConstraint(cmdModeConstraintV);
	ValueArg<tstring> cmdMode(TEXT("m"), TEXT("mode"), TEXT("image processing mode"),
		false, TEXT("noise_scale"), &cmdModeConstraint, cmd);

	std::vector<int> cmdNRLConstraintV;
	cmdNRLConstraintV.push_back(0);
	cmdNRLConstraintV.push_back(1);
	cmdNRLConstraintV.push_back(2);
	cmdNRLConstraintV.push_back(3);
	ValuesConstraint<int> cmdNRLConstraint(cmdNRLConstraintV);
	ValueArg<int> cmdNRLevel(TEXT("n"), TEXT("noise_level"), TEXT("noise reduction level"),
		false, 0, &cmdNRLConstraint, cmd);

	ValueArg<double> cmdScaleRatio(TEXT("s"), TEXT("scale_ratio"),
		TEXT("custom scale ratio"), false, 2.0, TEXT("double"), cmd);

	ValueArg<double> cmdScaleWidth(TEXT("w"), TEXT("scale_width"),
		TEXT("custom scale width"), false, 0, TEXT("double"), cmd);

	ValueArg<double> cmdScaleHeight(TEXT("h"), TEXT("scale_height"),
		TEXT("custom scale height"), false, 0, TEXT("double"), cmd);

	ValueArg<tstring> cmdModelPath(TEXT(""), TEXT("model_dir"),
		TEXT("path to custom model directory (don't append last / )"), false,
		TEXT("models/upconv_7_anime_style_art_rgb"), TEXT("string"), cmd);

	std::vector<tstring> cmdProcessConstraintV;
	cmdProcessConstraintV.push_back(TEXT("cpu"));
	cmdProcessConstraintV.push_back(TEXT("gpu"));
	cmdProcessConstraintV.push_back(TEXT("cudnn"));
	ValuesConstraint<tstring> cmdProcessConstraint(cmdProcessConstraintV);
	ValueArg<tstring> cmdProcess(TEXT("p"), TEXT("process"), TEXT("process mode"),
		false, TEXT("gpu"), &cmdProcessConstraint, cmd);

	ValueArg<int> cmdOutputQuality(TEXT("q"), TEXT("output_quality"),
		TEXT("output image quality"), false,
		-1, TEXT("int"), cmd);

	ValueArg<int> cmdOutputDepth(TEXT("d"), TEXT("output_depth"),
		TEXT("output image chaneel depth bit"), false,
		8, TEXT("int"), cmd);

	ValueArg<int> cmdCropSizeFile(TEXT("c"), TEXT("crop_size"),
		TEXT("input image split size"), false,
		128, TEXT("int"), cmd);

	ValueArg<int> cmdCropWidth(TEXT(""), TEXT("crop_w"),
		TEXT("input image split size(width)"), false,
		128, TEXT("int"), cmd);

	ValueArg<int> cmdCropHeight(TEXT(""), TEXT("crop_h"),
		TEXT("input image split size(height)"), false,
		128, TEXT("int"), cmd);

	ValueArg<int> cmdBatchSizeFile(TEXT("b"), TEXT("batch_size"),
		TEXT("input batch size"), false,
		1, TEXT("int"), cmd);

	ValueArg<int> cmdGPUNoFile(TEXT(""), TEXT("gpu"),
		TEXT("gpu device no"), false,
		0, TEXT("int"), cmd);

	std::vector<int> cmdTTAConstraintV;
	cmdTTAConstraintV.push_back(0);
	cmdTTAConstraintV.push_back(1);
	ValuesConstraint<int> cmdTTAConstraint(cmdTTAConstraintV);
	ValueArg<int> cmdTTALevel(TEXT("t"), TEXT("tta"), TEXT("8x slower and slightly high quality"),
		false, 0, &cmdTTAConstraint, cmd);

	// definition of command line argument : end

	Arg::enableIgnoreMismatched();

	// parse command line arguments
	try
	{
#ifdef WIN_UNICODE
		int nArgs = 0;
		LPTSTR *lplpszArgs = CommandLineToArgvW(GetCommandLine(), &nArgs);
		cmd.parse(nArgs, lplpszArgs);
		LocalFree(lplpszArgs);
#else
		cmd.parse(argc, argv);
#endif
	}
	catch (std::exception &e)
	{
		tprintf(TEXT("エラー: ") CHAR_STR_FORMAT TEXT("\n"), e.what());
		return 1;
	}

	boost::optional<double> ScaleRatio;
	boost::optional<int> ScaleWidth;
	boost::optional<int> ScaleHeight;

	int crop_w = cmdCropSizeFile.getValue();
	int crop_h = cmdCropSizeFile.getValue();

	if (cmdCropWidth.isSet())
		crop_w = cmdCropWidth.getValue();

	if (cmdCropHeight.isSet())
		crop_h = cmdCropHeight.getValue();

	if (cmdScaleWidth.getValue() > 0)
		ScaleWidth = cmdScaleWidth.getValue();
	if (cmdScaleHeight.getValue() > 0)
		ScaleHeight = cmdScaleHeight.getValue();

	if (cmdScaleWidth.getValue() == 0 && cmdScaleHeight.getValue() == 0)
		ScaleRatio = cmdScaleRatio.getValue();

	const boost::filesystem::path input_path(boost::filesystem::absolute((cmdInputFile.getValue())));

	tstring outputExt = cmdOutputFileExt.getValue();
	if (outputExt.length() > 0 && outputExt[0] != TEXT('.'))
		outputExt = TEXT(".") + outputExt;

	const std::string ModelName = Waifu2x::GetModelName(cmdModelPath.getValue());

	tstring tModelName;

#ifdef WIN_UNICODE
	{
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
		tModelName = cv.from_bytes(ModelName);
	}
#else
	tModelName = ModelName;
#endif

	const bool use_tta = cmdTTALevel.getValue() == 1;

	std::vector<std::pair<tstring, tstring>> file_paths;
	if (boost::filesystem::is_directory(input_path)) // input_pathがフォルダならそのディレクトリ以下の画像ファイルを一括変換
	{
		boost::filesystem::path output_path;

		if (cmdOutputFile.getValue() == TEXT("(auto)"))
		{
			// 「test」なら「test_noise_scale(Level1)(x2.000000)」みたいな感じにする

			tstring addstr(TEXT("("));
			addstr += tModelName;
			addstr += TEXT(")");

			const tstring &mode = cmdMode.getValue();

			addstr += TEXT("(") + mode + TEXT(")");

			if (mode.find(TEXT("noise")) != mode.npos || mode.find(TEXT("auto_scale")) != mode.npos)
				addstr += TEXT("(Level") + to_tstring(cmdNRLevel.getValue()) + TEXT(")");

			if (use_tta)
				addstr += TEXT("(tta)");
			if (mode.find(TEXT("scale")) != mode.npos)
			{
				if(ScaleRatio)
					addstr += TEXT("(x") + to_tstring(*ScaleRatio) + TEXT(")");
				else if (ScaleWidth && ScaleHeight)
					addstr += TEXT("(") + to_tstring(*ScaleWidth) + TEXT("x") + to_tstring(*ScaleHeight) + TEXT(")");
				else if (ScaleWidth)
					addstr += TEXT("(width ") + to_tstring(*ScaleWidth) + TEXT(")");
				else if (ScaleHeight)
					addstr += TEXT("(height ") + to_tstring(*ScaleHeight) + TEXT(")");
			}

			if (cmdOutputDepth.getValue() != 8)
				addstr += TEXT("(") + to_tstring(cmdOutputDepth.getValue()) + TEXT("bit)");

			output_path = input_path.branch_path() / (path_to_tstring(input_path.stem()) + addstr);
		}
		else
			output_path = cmdOutputFile.getValue();

		output_path = boost::filesystem::absolute(output_path);

		if (!boost::filesystem::exists(output_path))
		{
			if (!boost::filesystem::create_directory(output_path))
			{
				tprintf(TEXT("エラー: 出力フォルダ「%s」の作成に失敗しました\n"), path_to_tstring(output_path).c_str());
				return 1;
			}
		}

		std::vector<tstring> extList;
		{
			// input_extention_listを文字列の配列にする

			typedef boost::char_separator<TCHAR> char_separator;
			typedef boost::tokenizer<char_separator, tstring::const_iterator, tstring> tokenizer;

			char_separator sep(TEXT(":"), TEXT(""), boost::drop_empty_tokens);
			tokenizer tokens(cmdInputFileExt.getValue(), sep);

			for (tokenizer::iterator tok_iter = tokens.begin(); tok_iter != tokens.end(); ++tok_iter)
			{
				tstring ext(*tok_iter);
				std::transform(ext.begin(), ext.end(), ext.begin(), totlower);
				extList.push_back(TEXT(".") + ext);
			}
		}

		// 変換する画像の入力、出力パスを取得
		const auto func = [&extList, &input_path, &output_path, &outputExt, &file_paths](const boost::filesystem::path &path)
		{
			BOOST_FOREACH(const boost::filesystem::path& p, std::make_pair(boost::filesystem::recursive_directory_iterator(path),
				boost::filesystem::recursive_directory_iterator()))
			{
				if (boost::filesystem::is_directory(p))
				{
					const auto out_relative = relativePath(p, input_path);
					const auto out_absolute = output_path / out_relative;

					if (!boost::filesystem::exists(out_absolute))
					{
						if (!boost::filesystem::create_directory(out_absolute))
						{
							tprintf(TEXT("エラー: 出力フォルダ「%s」の作成に失敗しました\n"), path_to_tstring(out_absolute).c_str());
							return false;
						}
					}
				}
				else
				{
					tstring ext(path_to_tstring(p.extension()));
					std::transform(ext.begin(), ext.end(), ext.begin(), totlower);
					if (std::find(extList.begin(), extList.end(), ext) != extList.end())
					{
						const auto out_relative = relativePath(p, input_path);
						const auto out_absolute = output_path / out_relative;

						const auto out = path_to_tstring(out_absolute.branch_path() / out_absolute.stem()) + outputExt;

						file_paths.emplace_back(path_to_tstring(p), out);
					}
				}
			}

			return true;
		};

		if (!func(input_path))
			return 1;
	}
	else
	{
		tstring outputFileName = cmdOutputFile.getValue();

		if (outputFileName == TEXT("(auto)"))
		{
			// 「miku_small.png」なら「miku_small(noise_scale)(Level1)(x2.000000).png」みたいな感じにする

			outputFileName = cmdInputFile.getValue();
			const auto tailDot = outputFileName.find_last_of('.');
			outputFileName.erase(tailDot, outputFileName.length());

			tstring addstr(TEXT("("));
			addstr += tModelName;
			addstr += TEXT(")");

			const tstring &mode = cmdMode.getValue();

			addstr += TEXT("(") + mode + TEXT(")");

			if (mode.find(TEXT("noise")) != mode.npos || mode.find(TEXT("auto_scale")) != mode.npos)
				addstr += TEXT("(Level") + to_tstring(cmdNRLevel.getValue()) + TEXT(")");

			if (use_tta)
				addstr += TEXT("(tta)");
			if (mode.find(TEXT("scale")) != mode.npos)
			{
				if (ScaleRatio)
					addstr += TEXT("(x") + to_tstring(*ScaleRatio) + TEXT(")");
				else if (ScaleWidth && ScaleHeight)
					addstr += TEXT("(") + to_tstring(*ScaleWidth) + TEXT("x") + to_tstring(*ScaleHeight) + TEXT(")");
				else if (ScaleWidth)
					addstr += TEXT("(width ") + to_tstring(*ScaleWidth) + TEXT(")");
				else
					addstr += TEXT("(height ") + to_tstring(*ScaleHeight) + TEXT(")");
			}

			if (cmdOutputDepth.getValue() != 8)
				addstr += TEXT("(") + to_tstring(cmdOutputDepth.getValue()) + TEXT("bit)");

			outputFileName += addstr + outputExt;
		}

		file_paths.emplace_back(cmdInputFile.getValue(), outputFileName);
	}

	Waifu2x::eWaifu2xModelType mode;
	if (cmdMode.getValue() == TEXT("noise"))
		mode = Waifu2x::eWaifu2xModelTypeNoise;
	else if (cmdMode.getValue() == TEXT("scale"))
		mode = Waifu2x::eWaifu2xModelTypeScale;
	else if (cmdMode.getValue() == TEXT("noise_scale"))
		mode = Waifu2x::eWaifu2xModelTypeNoiseScale;
	else if (cmdMode.getValue() == TEXT("auto_scale"))
		mode = Waifu2x::eWaifu2xModelTypeAutoScale;

	Waifu2x::eWaifu2xError ret;
	Waifu2x w;

#ifdef WIN_UNICODE
	std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
	const std::string sProcess = cv.to_bytes(cmdProcess.getValue());
#else
	const std::string sProcess = cmdProcess.getValue();
#endif

	ret = w.Init(mode, cmdNRLevel.getValue(), cmdModelPath.getValue(), sProcess, cmdGPUNoFile.getValue());
	switch (ret)
	{
	case Waifu2x::eWaifu2xError_InvalidParameter:
		tprintf(TEXT("エラー: パラメータが不正です\n"));
		return 1;
	case Waifu2x::eWaifu2xError_FailedOpenModelFile:
		tprintf(TEXT("エラー: モデルファイルが開けませんでした\n"));
		return 1;
	case Waifu2x::eWaifu2xError_FailedParseModelFile:
		tprintf(TEXT("エラー: モデルファイルが壊れています\n"));
		return 1;
	case Waifu2x::eWaifu2xError_FailedConstructModel:
		tprintf(TEXT("エラー: ネットワークの構築に失敗しました\n"));
		return 1;
	}

	bool isError = false;
	for (const auto &p : file_paths)
	{
		const Waifu2x::eWaifu2xError ret = w.waifu2x(p.first, p.second, ScaleRatio, ScaleWidth, ScaleHeight, nullptr,
			crop_w, crop_h,
			cmdOutputQuality.getValue() == -1 ? boost::optional<int>() : cmdOutputQuality.getValue(), cmdOutputDepth.getValue(), use_tta, cmdBatchSizeFile.getValue());
		if (ret != Waifu2x::eWaifu2xError_OK)
		{
			switch (ret)
			{
			case Waifu2x::eWaifu2xError_InvalidParameter:
				tprintf(TEXT("エラー: パラメータが不正です\n"));
				break;
			case Waifu2x::eWaifu2xError_FailedOpenInputFile:
				tprintf(TEXT("エラー: 入力画像「%s」が開けませんでした\n"), p.first.c_str());
				break;
			case Waifu2x::eWaifu2xError_FailedOpenOutputFile:
				tprintf(TEXT("エラー: 出力画像「%s」が書き込めませんでした\n"), p.second.c_str());
				break;
			case Waifu2x::eWaifu2xError_FailedProcessCaffe:
				tprintf(TEXT("エラー: 補間処理に失敗しました\n"));
				break;
			}

			isError = true;
		}
	}

	if (isError)
	{
		tprintf(TEXT("変換に失敗したファイルがあります\n"));
		return 1;
	}

	tprintf(TEXT("変換に成功しました\n"));

	Waifu2x::quit_liblary();

	return 0;
}
