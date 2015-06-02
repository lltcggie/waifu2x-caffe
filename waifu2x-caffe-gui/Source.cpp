#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <Commctrl.h>
#include <tchar.h>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include "resource.h"
#include "../common/waifu2x.h"

#include "CDialog.h"
#include "CControl.h"

#define WM_FAILD_CREATE_DIR (WM_APP + 5)
#define WM_ON_WAIFU2X_ERROR (WM_APP + 6)
#define WM_END_THREAD (WM_APP + 7)

const size_t AR_PATH_MAX(1024);


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

// ダイアログ用
class DialogEvent
{
private:
	HWND dh;

	std::string input_str;
	std::string output_str;
	std::string mode;
	int noise_level;
	double scale_ratio;
	std::string process;
	std::string outputExt;
	std::string inputFileExt;

	int crop_size;
	int batch_size;

	std::thread processThread;
	std::atomic_bool cancelFlag;

	std::string autoSetAddName;
	bool isLastError;

	std::string logMessage;

	std::string usedProcess;
	std::chrono::system_clock::duration cuDNNCheckTime;
	std::chrono::system_clock::duration InitTime;
	std::chrono::system_clock::duration ProcessTime;

private:
	std::string AddName() const
	{
		std::string addstr("(" + mode + ")");

		if (mode.find("noise") != mode.npos || mode.find("auto_scale") != mode.npos)
			addstr += "(Level" + std::to_string(noise_level) + ")";
		if (mode.find("scale") != mode.npos)
			addstr += "(x" + std::to_string(scale_ratio) + ")";

		return addstr;
	}

	bool SyncMember()
	{
		bool ret = true;

		{
			char buf[AR_PATH_MAX] = "";
			GetWindowTextA(GetDlgItem(dh, IDC_EDIT_INPUT), buf, _countof(buf));
			buf[_countof(buf) - 1] = '\0';

			input_str = buf;
		}

		{
			char buf[AR_PATH_MAX] = "";
			GetWindowTextA(GetDlgItem(dh, IDC_EDIT_OUTPUT), buf, _countof(buf));
			buf[_countof(buf) - 1] = '\0';

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

		{
			char buf[AR_PATH_MAX] = "";
			GetWindowTextA(GetDlgItem(dh, IDC_EDIT_SCALE_RATIO), buf, _countof(buf));
			buf[_countof(buf) - 1] = '\0';

			char *ptr = nullptr;
			scale_ratio = strtod(buf, &ptr);
			if (!ptr || *ptr != '\0')
			{
				scale_ratio = 2.0;
				ret = false;

				MessageBox(dh, TEXT("拡大率は0.0より大きい正数である必要があります"), TEXT("エラー"), MB_OK | MB_ICONERROR);
			}
		}

		{
			char buf[AR_PATH_MAX] = "";
			GetWindowTextA(GetDlgItem(dh, IDC_EDIT_OUT_EXT), buf, _countof(buf));
			buf[_countof(buf) - 1] = '\0';

			outputExt = buf;
			if (outputExt.length() > 0 && outputExt[0] != '.')
				outputExt = "." + outputExt;
		}

		if (SendMessage(GetDlgItem(dh, IDC_RADIO_MODE_CPU), BM_GETCHECK, 0, 0))
			process = "cpu";
		else
			process = "gpu";

		{
			char buf[AR_PATH_MAX] = "";
			GetWindowTextA(GetDlgItem(dh, IDC_EDIT_INPUT_EXT_LIST), buf, _countof(buf));
			buf[_countof(buf) - 1] = '\0';

			inputFileExt = buf;
		}

		{
			char buf[AR_PATH_MAX] = "";
			GetWindowTextA(GetDlgItem(dh, IDC_EDIT_CROP_SIZE), buf, _countof(buf));
			buf[_countof(buf) - 1] = '\0';

			char *ptr = nullptr;
			crop_size = strtol (buf, &ptr, 10);
			if (!ptr || *ptr != '\0')
			{
				crop_size = 128;
				ret = false;

				MessageBox(dh, TEXT("分割サイズは0より大きい整数である必要があります"), TEXT("エラー"), MB_OK | MB_ICONERROR);
			}
		}

		return ret;
	}

	void ProcessWaifu2x()
	{
		const boost::filesystem::path input_path(boost::filesystem::absolute(input_str));

		std::vector<std::pair<std::string, std::string>> file_paths;
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

			std::vector<std::string> extList;
			{
				// input_extention_listを文字列の配列にする

				typedef boost::char_separator<char> char_separator;
				typedef boost::tokenizer<char_separator> tokenizer;

				char_separator sep(":", "", boost::drop_empty_tokens);
				tokenizer tokens(inputFileExt, sep);

				for (tokenizer::iterator tok_iter = tokens.begin(); tok_iter != tokens.end(); ++tok_iter)
					extList.push_back("." + *tok_iter);
			}

			// 変換する画像の入力、出力パスを取得
			const auto func = [this, &extList, &input_path, &output_path, &file_paths](const boost::filesystem::path &path)
			{
				BOOST_FOREACH(const boost::filesystem::path& p, std::make_pair(boost::filesystem::recursive_directory_iterator(path),
					boost::filesystem::recursive_directory_iterator()))
				{
					if (!boost::filesystem::is_directory(p) && std::find(extList.begin(), extList.end(), p.extension().string()) != extList.end())
					{
						const auto out_relative = relativePath(p, input_path);
						const auto out_absolute = output_path / out_relative;

						const auto out = (out_absolute.branch_path() / out_absolute.stem()).string() + outputExt;

						file_paths.emplace_back(p.string(), out);
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
			file_paths.emplace_back(input_str, output_str);

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

		Waifu2x w;
		ret = w.init(__argc, __argv, mode, noise_level, scale_ratio, "models", process, crop_size, batch_size);
		if(ret != Waifu2x::eWaifu2xError_OK)
			SendMessage(dh, WM_ON_WAIFU2X_ERROR, (WPARAM)&ret, 0);
		else
		{
			const auto InitEndTime = std::chrono::system_clock::now();

			for (const auto &p : file_paths)
			{
				ret = w.waifu2x(p.first, p.second, [this]()
				{
					return cancelFlag;
				});

				if (ret != Waifu2x::eWaifu2xError_OK)
				{
					SendMessage(dh, WM_ON_WAIFU2X_ERROR, (WPARAM)&ret, (LPARAM)&p);

					if (ret == Waifu2x::eWaifu2xError_Cancel)
						break;
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

	void ReplaceAddString()
	{
		SyncMember();

		const boost::filesystem::path output_path(output_str);
		std::string stem = output_path.stem().string();
		if (stem.length() > 0 && stem.length() >= autoSetAddName.length())
		{
			const std::string base = stem.substr(0, stem.length() - autoSetAddName.length());
			stem.erase(0, base.length());
			if (stem == autoSetAddName)
			{
				const std::string addstr(AddName());
				autoSetAddName = addstr;

				boost::filesystem::path new_out_path = output_path.branch_path() / (base + addstr + outputExt);

				SetWindowTextA(GetDlgItem(dh, IDC_EDIT_OUTPUT), new_out_path.string().c_str());
			}
		}
	}

	void AddLogMessage(const char *msg)
	{
		if (logMessage.length() == 0)
			logMessage += msg;
		else
			logMessage += std::string("\r\n") + msg;

		SetWindowTextA(GetDlgItem(dh, IDC_EDIT_LOG), logMessage.c_str());
	}

	void Waifu2xTime()
	{
		char msg[1024 * 2];
		char *ptr = msg;

		{
			std::string p(usedProcess);
			if (p == "cpu")
				p = "CPU";
			else if (p == "gpu")
				p = "GPU";
			else if (p == "cudnn")
				p = "cuDNN";

			ptr += sprintf(ptr, "使用プロセッサーモード: %s\r\n", p.c_str());
		}

		{
			uint64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(ProcessTime).count();
			const int msec = t % 1000; t /= 1000;
			const int sec = t % 60; t /= 60;
			const int min = t % 60; t /= 60;
			const int hour = (int)t;
			ptr += sprintf(ptr, "処理時間: %02d:%02d:%02d.%d\r\n", hour, min, sec, msec);
		}

		{
			uint64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(InitTime).count();
			const int msec = t % 1000; t /= 1000;
			const int sec = t % 60; t /= 60;
			const int min = t % 60; t /= 60;
			const int hour = (int)t;
			ptr += sprintf(ptr, "初期化時間: %02d:%02d:%02d.%d\r\n", hour, min, sec, msec);
		}

		if (process == "gpu" || process == "cudnn")
		{
			uint64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(cuDNNCheckTime).count();
			const int msec = t % 1000; t /= 1000;
			const int sec = t % 60; t /= 60;
			const int min = t % 60; t /= 60;
			const int hour = (int)t;
			ptr += sprintf(ptr, "cuDNNチェック時間: %02d:%02d:%02d.%d", hour, min, sec, msec);
		}

		AddLogMessage(msg);
	}

public:
	DialogEvent() : dh(nullptr), mode("noise_scale"), noise_level(1), scale_ratio(2.0), process("gpu"), outputExt("png"), inputFileExt("png:jpg:jpeg:tif:tiff:bmp"),
		crop_size(128), batch_size(1), isLastError(false)
	{
	}

	void Exec(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		if (processThread.joinable())
			return;

		SyncMember();

		if (input_str.length() == 0)
		{
			MessageBox(dh, TEXT("入力パスを指定して下さい"), TEXT("エラー"), MB_OK | MB_ICONERROR);
			return;
		}

		if (output_str.length() == 0)
		{
			MessageBox(dh, TEXT("出力パスを指定して下さい"), TEXT("エラー"), MB_OK | MB_ICONERROR);
			return;
		}

		if (outputExt.length() == 0)
		{
			MessageBox(dh, TEXT("出力拡張子を指定して下さい"), TEXT("エラー"), MB_OK | MB_ICONERROR);
			return;
		}

		SendMessage(GetDlgItem(dh, IDC_PROGRESS), PBM_SETPOS, 0, 0);
		cancelFlag = false;
		isLastError = false;

		processThread = std::thread(std::bind(&DialogEvent::ProcessWaifu2x, this));

		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CANCEL), TRUE);
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_EXEC), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CHECK_CUDNN), FALSE);

		SetWindowTextA(GetDlgItem(hWnd, IDC_EDIT_LOG), "");
		logMessage.clear();
	}

	void WaitThreadExit(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		processThread.join();
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CANCEL), FALSE);
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_EXEC), TRUE);
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CHECK_CUDNN), TRUE);

		if (!isLastError)
		{
			if (!cancelFlag)
				AddLogMessage("変換に成功しました");

			Waifu2xTime();
			MessageBeep(MB_ICONASTERISK);
		}
		else
			MessageBoxA(dh, "エラーが発生しました", "エラー", MB_OK | MB_ICONERROR);
	}

	void OnDialogEnd(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		if (!processThread.joinable())
			PostQuitMessage(0);
		else
			MessageBeep(MB_ICONEXCLAMATION);
	}

	void OnFaildCreateDir(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		const boost::filesystem::path *p = (const boost::filesystem::path *)wParam;

		// 出力フォルダ「%s」の作成に失敗しました\n", out_absolute.string().c_str());
		std::wstring msg(L"出力フォルダ\r\n「");
		msg += p->wstring();
		msg += L"」\r\nの作成に失敗しました";

		MessageBox(dh, msg.c_str(), TEXT("エラー"), MB_OK | MB_ICONERROR);

		isLastError = true;
	}

	void OnWaifu2xError(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		const Waifu2x::eWaifu2xError ret = *(const Waifu2x::eWaifu2xError *)wParam;

		if (ret != Waifu2x::eWaifu2xError_OK)
		{
			char msg[1024] = "";

			if (lParam == 0)
			{
				switch (ret)
				{
				case Waifu2x::eWaifu2xError_Cancel:
					sprintf(msg, "キャンセルされました");
					break;
				case Waifu2x::eWaifu2xError_InvalidParameter:
					sprintf(msg, "パラメータが不正です");
					break;
				case Waifu2x::eWaifu2xError_FailedOpenModelFile:
					sprintf(msg, "モデルファイルが開けませんでした");
					break;
				case Waifu2x::eWaifu2xError_FailedParseModelFile:
					sprintf(msg, "モデルファイルが壊れています");
					break;
				case Waifu2x::eWaifu2xError_FailedConstructModel:
					sprintf(msg, "ネットワークの構築に失敗しました");
					break;
				}
			}
			else
			{
				const auto &fp = *(const std::pair<std::string, std::string> *)lParam;

				switch (ret)
				{
				case Waifu2x::eWaifu2xError_Cancel:
					sprintf(msg, "キャンセルされました");
					break;
				case Waifu2x::eWaifu2xError_InvalidParameter:
					sprintf(msg, "パラメータが不正です");
					break;
				case Waifu2x::eWaifu2xError_FailedOpenInputFile:
					sprintf(msg, "入力画像「%s」が開けませんでした", fp.first.c_str());
					break;
				case Waifu2x::eWaifu2xError_FailedOpenOutputFile:
					sprintf(msg, "出力画像を「%s」に書き込めませんでした", fp.second.c_str());
					break;
				case Waifu2x::eWaifu2xError_FailedProcessCaffe:
					sprintf(msg, "補間処理に失敗しました");
					break;
				}
			}

			AddLogMessage(msg);

			if (ret != Waifu2x::eWaifu2xError_Cancel)
				isLastError = true;
		}
	}

	void Create(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		dh = hWnd;

		SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_NOISE_SCALE), BM_SETCHECK, BST_CHECKED, 0);
		SendMessage(GetDlgItem(hWnd, IDC_RADIONOISE_LEVEL1), BM_SETCHECK, BST_CHECKED, 0);
		SendMessage(GetDlgItem(hWnd, IDC_RADIO_MODE_GPU), BM_SETCHECK, BST_CHECKED, 0);

		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CANCEL), FALSE);

		char text[] = "2.00";
		SetWindowTextA(GetDlgItem(hWnd, IDC_EDIT_SCALE_RATIO), text);
		SetWindowTextA(GetDlgItem(hWnd, IDC_EDIT_OUT_EXT), outputExt.c_str());
		SetWindowTextA(GetDlgItem(hWnd, IDC_EDIT_INPUT_EXT_LIST), inputFileExt.c_str());
		SetWindowTextA(GetDlgItem(hWnd, IDC_EDIT_CROP_SIZE), std::to_string(crop_size).c_str());
	}

	void Cancel(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		cancelFlag = true;
		EnableWindow(GetDlgItem(dh, IDC_BUTTON_CANCEL), FALSE);
	}

	void RadioButtom(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		ReplaceAddString();
	}

	void CheckCUDNN(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData)
	{
		if (Waifu2x::can_use_cuDNN())
			MessageBox(dh, TEXT("cuDNNが使えます"), TEXT("結果"), MB_OK | MB_ICONINFORMATION);
		else
			MessageBox(dh, TEXT("cuDNNは使えません"), TEXT("結果"), MB_OK | MB_ICONERROR);
	}

	// ここで渡されるhWndはIDC_EDITのHWND(コントロールのイベントだから)
	LRESULT DropInput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData)
	{
		char szTmp[AR_PATH_MAX];

		// ドロップされたファイル数を取得
		UINT FileNum = DragQueryFileA((HDROP)wParam, 0xFFFFFFFF, szTmp, _countof(szTmp));
		if (FileNum >= 1)
		{
			DragQueryFileA((HDROP)wParam, 0, szTmp, _countof(szTmp));

			boost::filesystem::path path(szTmp);

			if (!SyncMember())
				return 0L;

			if (boost::filesystem::is_directory(path))
			{
				HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

				const std::string addstr(AddName());
				autoSetAddName = AddName();

				auto str = (path.branch_path() / (path.stem().string() + addstr)).string();

				SetWindowTextA(ho, str.c_str());

				SetWindowTextA(hWnd, szTmp);
			}
			else
			{
				HWND ho = GetDlgItem(dh, IDC_EDIT_OUTPUT);

				std::string outputFileName = szTmp;

				const auto tailDot = outputFileName.find_last_of('.');
				outputFileName.erase(tailDot, outputFileName.length());

				const std::string addstr(AddName());
				autoSetAddName = addstr;

				outputFileName += addstr + outputExt;

				SetWindowTextA(ho, outputFileName.c_str());

				SetWindowTextA(hWnd, szTmp);
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
};


int WINAPI WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR     lpCmdLine,
	int       nCmdShow)
{
	// CDialogクラスでダイアログを作成する
	CDialog cDialog;
	CDialog cDialog2;
	// IDC_EDITのサブクラス
	CControl cControlInput(IDC_EDIT_INPUT);
	CControl cControlOutput(IDC_EDIT_OUTPUT);
	CControl cControlScale(IDC_EDIT_SCALE_RATIO);
	CControl cControlOutExt(IDC_EDIT_OUT_EXT);

	// 登録する関数がまとめられたクラス
	// グローバル関数を使えばクラスにまとめる必要はないがこの方法が役立つこともあるはず
	DialogEvent cDialogEvent;

	// クラスの関数を登録する場合

	// IDC_EDITにWM_DROPFILESが送られてきたときに実行する関数の登録
	cControlInput.SetEventCallBack(SetClassCustomFunc(DialogEvent::DropInput, &cDialogEvent), NULL, WM_DROPFILES);
	cControlOutput.SetEventCallBack(SetClassCustomFunc(DialogEvent::DropOutput, &cDialogEvent), NULL, WM_DROPFILES);
	cControlScale.SetEventCallBack(SetClassCustomFunc(DialogEvent::TextInput, &cDialogEvent), NULL, WM_CHAR);
	cControlOutExt.SetEventCallBack(SetClassCustomFunc(DialogEvent::TextInput, &cDialogEvent), NULL, WM_CHAR);

	// コントロールのサブクラスを登録
	cDialog.AddControl(&cControlInput);
	cDialog.AddControl(&cControlOutput);
	cDialog.AddControl(&cControlScale);
	cDialog.AddControl(&cControlOutExt);

	// 各コントロールのイベントで実行する関数の登録
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::Exec, &cDialogEvent), NULL, IDC_BUTTON_EXEC);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::Cancel, &cDialogEvent), NULL, IDC_BUTTON_CANCEL);

	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::RadioButtom, &cDialogEvent), NULL, IDC_RADIO_MODE_NOISE);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::RadioButtom, &cDialogEvent), NULL, IDC_RADIO_MODE_SCALE);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::RadioButtom, &cDialogEvent), NULL, IDC_RADIO_MODE_NOISE_SCALE);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::RadioButtom, &cDialogEvent), NULL, IDC_RADIO_AUTO_SCALE);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::RadioButtom, &cDialogEvent), NULL, IDC_RADIONOISE_LEVEL1);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::RadioButtom, &cDialogEvent), NULL, IDC_RADIONOISE_LEVEL2);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::RadioButtom, &cDialogEvent), NULL, IDC_RADIO_MODE_CPU);
	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::RadioButtom, &cDialogEvent), NULL, IDC_RADIO_MODE_GPU);

	cDialog.SetCommandCallBack(SetClassFunc(DialogEvent::CheckCUDNN, &cDialogEvent), NULL, IDC_BUTTON_CHECK_CUDNN);

	// ダイアログのイベントで実行する関数の登録
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::Create, &cDialogEvent), NULL, WM_INITDIALOG);
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::OnDialogEnd, &cDialogEvent), NULL, WM_CLOSE);
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::OnFaildCreateDir, &cDialogEvent), NULL, WM_FAILD_CREATE_DIR);
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::OnWaifu2xError, &cDialogEvent), NULL, WM_ON_WAIFU2X_ERROR);
	cDialog.SetEventCallBack(SetClassFunc(DialogEvent::WaitThreadExit, &cDialogEvent), NULL, WM_END_THREAD);

	// ダイアログを表示
	cDialog.DoModal(hInstance, IDD_DIALOG);

	return 0;
}
