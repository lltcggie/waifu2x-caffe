#include <stdio.h>
#include <algorithm>
#include <tclap/CmdLine.h>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <functional>
#include <boost/tokenizer.hpp>
#include <boost/tokenizer.hpp>
#include <glog/logging.h>
#include "../common/waifu2x.h"


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
	Waifu2x::init_liblary();

	// Caffeのエラーでないログを保存しないようにする
	google::SetLogDestination(google::INFO, "");
	google::SetLogDestination(google::WARNING, "");

	// Caffeのエラーログを「error_log_～」に出力
	google::SetLogDestination(google::ERROR, "error_log_");
	google::SetLogDestination(google::FATAL, "error_log_");

	// definition of command line arguments
	TCLAP::CmdLine cmd("waifu2x reimplementation using Caffe", ' ', "1.0.0");

	TCLAP::ValueArg<std::string> cmdInputFile("i", "input_path",
		"path to input image file", true, "",
		"string", cmd);

	TCLAP::ValueArg<std::string> cmdOutputFile("o", "output_path",
		"path to output image file (when input_path is folder, output_path must be folder)", false,
		"(auto)", "string", cmd);

	TCLAP::ValueArg<std::string> cmdInputFileExt("l", "input_extention_list",
		"extention to input image file when input_path is folder", false, "png:jpg:jpeg:tif:tiff:bmp:tga",
		"string", cmd);

	TCLAP::ValueArg<std::string> cmdOutputFileExt("e", "output_extention",
		"extention to output image file when output_path is (auto) or input_path is folder", false,
		"png", "string", cmd);

	std::vector<std::string> cmdModeConstraintV;
	cmdModeConstraintV.push_back("noise");
	cmdModeConstraintV.push_back("scale");
	cmdModeConstraintV.push_back("noise_scale");
	cmdModeConstraintV.push_back("auto_scale");
	TCLAP::ValuesConstraint<std::string> cmdModeConstraint(cmdModeConstraintV);
	TCLAP::ValueArg<std::string> cmdMode("m", "mode", "image processing mode",
		false, "noise_scale", &cmdModeConstraint, cmd);

	std::vector<int> cmdNRLConstraintV;
	cmdNRLConstraintV.push_back(1);
	cmdNRLConstraintV.push_back(2);
	cmdNRLConstraintV.push_back(3);
	TCLAP::ValuesConstraint<int> cmdNRLConstraint(cmdNRLConstraintV);
	TCLAP::ValueArg<int> cmdNRLevel("n", "noise_level", "noise reduction level",
		false, 1, &cmdNRLConstraint, cmd);

	TCLAP::ValueArg<double> cmdScaleRatio("s", "scale_ratio",
		"custom scale ratio", false, 2.0, "double", cmd);

	TCLAP::ValueArg<double> cmdScaleWidth("w", "scale_width",
		"custom scale width", false, 0, "double", cmd);

	TCLAP::ValueArg<double> cmdScaleHeight("h", "scale_height",
		"custom scale height", false, 0, "double", cmd);

	TCLAP::ValueArg<std::string> cmdModelPath("", "model_dir",
		"path to custom model directory (don't append last / )", false,
		"models/anime_style_art_rgb", "string", cmd);

	std::vector<std::string> cmdProcessConstraintV;
	cmdProcessConstraintV.push_back("cpu");
	cmdProcessConstraintV.push_back("gpu");
	cmdProcessConstraintV.push_back("cudnn");
	TCLAP::ValuesConstraint<std::string> cmdProcessConstraint(cmdProcessConstraintV);
	TCLAP::ValueArg<std::string> cmdProcess("p", "process", "process mode",
		false, "gpu", &cmdProcessConstraint, cmd);

	TCLAP::ValueArg<int> cmdOutputQuality("q", "output_quality",
		"output image quality", false,
		-1, "int", cmd);

	TCLAP::ValueArg<int> cmdOutputDepth("d", "output_depth",
		"output image chaneel depth bit", false,
		8, "int", cmd);

	TCLAP::ValueArg<int> cmdCropSizeFile("c", "crop_size",
		"input image split size", false,
		128, "int", cmd);

	TCLAP::ValueArg<int> cmdBatchSizeFile("b", "batch_size",
		"input batch size", false,
		1, "int", cmd);

	std::vector<int> cmdTTAConstraintV;
	cmdTTAConstraintV.push_back(0);
	cmdTTAConstraintV.push_back(1);
	TCLAP::ValuesConstraint<int> cmdTTAConstraint(cmdTTAConstraintV);
	TCLAP::ValueArg<int> cmdTTALevel("t", "tta", "8x slower and slightly high quality",
		false, 0, &cmdTTAConstraint, cmd);

	// definition of command line argument : end

	TCLAP::Arg::enableIgnoreMismatched();

	// parse command line arguments
	try
	{
		cmd.parse(argc, argv);
	}
	catch (std::exception &e)
	{
		printf("エラー: %s\n", e.what());
		return 1;
	}

	boost::optional<double> ScaleRatio;
	boost::optional<int> ScaleWidth;
	boost::optional<int> ScaleHeight;

	int valid_num = 0;
	if (cmdScaleWidth.getValue() > 0)
		valid_num++;
	if (cmdScaleHeight.getValue() > 0)
		valid_num++;

	if (valid_num > 1)
	{
		printf("エラー: scale_widthとscale_heightは同時に指定できません\n");
		return 1;
	}

	if (cmdScaleWidth.getValue() > 0)
		ScaleWidth = cmdScaleWidth.getValue();
	else if (cmdScaleHeight.getValue() > 0)
		ScaleHeight = cmdScaleHeight.getValue();
	else
		ScaleRatio = cmdScaleRatio.getValue();

	const boost::filesystem::path input_path(boost::filesystem::absolute((cmdInputFile.getValue())));

	std::string outputExt = cmdOutputFileExt.getValue();
	if (outputExt.length() > 0 && outputExt[0] != '.')
		outputExt = "." + outputExt;

	const bool use_tta = cmdTTALevel.getValue() == 1;

	std::vector<std::pair<std::string, std::string>> file_paths;
	if (boost::filesystem::is_directory(input_path)) // input_pathがフォルダならそのディレクトリ以下の画像ファイルを一括変換
	{
		boost::filesystem::path output_path;

		if (cmdOutputFile.getValue() == "(auto)")
		{
			// 「test」なら「test_noise_scale(Level1)(x2.000000)」みたいな感じにする

			std::string addstr("_" + cmdMode.getValue());

			const std::string &mode = cmdMode.getValue();
			if (mode.find("noise") != mode.npos || mode.find("auto_scale") != mode.npos)
				addstr += "(Level" + std::to_string(cmdNRLevel.getValue()) + ")";
			if (use_tta)
				addstr += "(tta)";
			if (mode.find("scale") != mode.npos)
			{
				if(ScaleRatio)
					addstr += "(x" + std::to_string(*ScaleRatio) + ")";
				else if (ScaleWidth)
					addstr += "(width " + std::to_string(*ScaleWidth) + ")";
				else
					addstr += "(height " + std::to_string(*ScaleHeight) + ")";
			}

			output_path = input_path.branch_path() / (input_path.stem().string() + addstr);
		}
		else
			output_path = cmdOutputFile.getValue();

		output_path = boost::filesystem::absolute(output_path);

		if (!boost::filesystem::exists(output_path))
		{
			if (!boost::filesystem::create_directory(output_path))
			{
				printf("エラー: 出力フォルダ「%s」の作成に失敗しました\n", output_path.string().c_str());
				return 1;
			}
		}

		std::vector<std::string> extList;
		{
			// input_extention_listを文字列の配列にする

			typedef boost::char_separator<char> char_separator;
			typedef boost::tokenizer<char_separator> tokenizer;

			char_separator sep(":", "", boost::drop_empty_tokens);
			tokenizer tokens(cmdInputFileExt.getValue(), sep);

			for (tokenizer::iterator tok_iter = tokens.begin(); tok_iter != tokens.end(); ++tok_iter)
			{
				std::string ext(*tok_iter);
				std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
				extList.push_back("." + ext);
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
							printf("エラー: 出力フォルダ「%s」の作成に失敗しました\n", out_absolute.string().c_str());
							return false;
						}
					}
				}
				else
				{
					std::string ext(p.extension().string());
					std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
					if (std::find(extList.begin(), extList.end(), ext) != extList.end())
					{
						const auto out_relative = relativePath(p, input_path);
						const auto out_absolute = output_path / out_relative;

						const auto out = (out_absolute.branch_path() / out_absolute.stem()).string() + outputExt;

						file_paths.emplace_back(p.string(), out);
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
		std::string outputFileName = cmdOutputFile.getValue();

		if (outputFileName == "(auto)")
		{
			// 「miku_small.png」なら「miku_small(noise_scale)(Level1)(x2.000000).png」みたいな感じにする

			outputFileName = cmdInputFile.getValue();
			const auto tailDot = outputFileName.find_last_of('.');
			outputFileName.erase(tailDot, outputFileName.length());
			outputFileName = outputFileName + "(" + cmdMode.getValue() + ")";
			std::string &mode = cmdMode.getValue();
			if (mode.find("noise") != mode.npos || mode.find("auto_scale") != mode.npos)
				outputFileName = outputFileName + "(Level" + std::to_string(cmdNRLevel.getValue()) + ")";
			if (use_tta)
				outputFileName += "(tta)";
			if (mode.find("scale") != mode.npos)
				outputFileName = outputFileName + "(x" + std::to_string(cmdScaleRatio.getValue()) + ")";
			outputFileName += outputExt;
		}

		file_paths.emplace_back(cmdInputFile.getValue(), outputFileName);
	}

	Waifu2x::eWaifu2xError ret;
	Waifu2x w;
	ret = w.init(argc, argv, cmdMode.getValue(), cmdNRLevel.getValue(), ScaleRatio, ScaleWidth, ScaleHeight, cmdModelPath.getValue(), cmdProcess.getValue(),
		cmdOutputQuality.getValue() == -1 ? boost::optional<int>() : cmdOutputQuality.getValue(), cmdOutputDepth.getValue(), use_tta, cmdCropSizeFile.getValue(),
		cmdBatchSizeFile.getValue());
	switch (ret)
	{
	case Waifu2x::eWaifu2xError_InvalidParameter:
		printf("エラー: パラメータが不正です\n");
		return 1;
	case Waifu2x::eWaifu2xError_FailedOpenModelFile:
		printf("エラー: モデルファイルが開けませんでした\n");
		return 1;
	case Waifu2x::eWaifu2xError_FailedParseModelFile:
		printf("エラー: モデルファイルが壊れています\n");
		return 1;
	case Waifu2x::eWaifu2xError_FailedConstructModel:
		printf("エラー: ネットワークの構築に失敗しました\n");
		return 1;
	}

	bool isError = false;
	for (const auto &p : file_paths)
	{
		const Waifu2x::eWaifu2xError ret = w.waifu2x(p.first, p.second);
		if (ret != Waifu2x::eWaifu2xError_OK)
		{
			switch (ret)
			{
			case Waifu2x::eWaifu2xError_InvalidParameter:
				printf("エラー: パラメータが不正です\n");
				break;
			case Waifu2x::eWaifu2xError_FailedOpenInputFile:
				printf("エラー: 入力画像「%s」が開けませんでした\n", p.first.c_str());
				break;
			case Waifu2x::eWaifu2xError_FailedOpenOutputFile:
				printf("エラー: 出力画像「%s」が書き込めませんでした\n", p.second.c_str());
				break;
			case Waifu2x::eWaifu2xError_FailedProcessCaffe:
				printf("エラー: 補間処理に失敗しました\n");
				break;
			}

			isError = true;
		}
	}

	if (isError)
	{
		printf("変換に失敗したファイルがあります\n");
		return 1;
	}

	printf("変換に成功しました\n");

	Waifu2x::quit_liblary();

	return 0;
}
