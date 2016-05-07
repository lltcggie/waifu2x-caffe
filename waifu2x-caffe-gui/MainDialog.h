#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <string>
#include <thread>
#include <atomic>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include "resource.h"
#include "tstring.h"
#include "LangStringList.h"

#undef ERROR

#define WM_FAILD_CREATE_DIR (WM_APP + 5)
#define WM_ON_WAIFU2X_ERROR (WM_APP + 6)
#define WM_ON_WAIFU2X_NO_OVERWRITE (WM_APP + 8)
#define WM_END_THREAD (WM_APP + 7)


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

	tstring output_dir;

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

	tstring tAutoMode;

	bool isArgStartAuto;
	bool isArgStartSuccessFinish;
	bool isOutputNoOverwrite;
	tstring tInputDirFix;
	tstring tOutputDirFix;

	bool isNotSaveParam;

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

	tstring AddName() const;

	bool SyncMember(const bool NotSyncCropSize, const bool silent = false);

	void SetCropSizeList(const boost::filesystem::path &input_path);

	static boost::filesystem::path GetFileName(const boost::filesystem::path &input_path)
	{
		if (boost::filesystem::is_directory(input_path))
			return input_path.stem();
		else
			return input_path.filename();
	}

	void ProcessWaifu2x();

	void ReplaceAddString();

	void AddLogMessage(const TCHAR *msg);

	void Waifu2xTime();

	void SaveIni(const bool isSyncMember = true);

	// 入力パスを選択する
	static UINT_PTR CALLBACK OFNHookProcIn(
		_In_  HWND hdlg,
		_In_  UINT uiMsg,
		_In_  WPARAM wParam,
		_In_  LPARAM lParam
	);

	// 出力パスを選択する
	static UINT_PTR CALLBACK OFNHookProcOut(
		_In_  HWND hdlg,
		_In_  UINT uiMsg,
		_In_  WPARAM wParam,
		_In_  LPARAM lParam
	);

public:
	DialogEvent();

	void Exec(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void WaitThreadExit(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void Timer(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void OnDialogEnd(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void OnFaildCreateDir(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void OnWaifu2xError(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void OnWaifu2xNoOverwrite(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void SetWindowTextLang();

	void SetDepthAndQuality(const bool SetDefaultQuality = true);

	void Create(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void Cancel(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void UpdateAddString(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void OnModeChange(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void ScaleRadio(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void CheckCUDNN(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	LRESULT OnSetInputFilePath(const TCHAR *tPath);

	LRESULT OnSetInputFilePath();

	LRESULT OnSetOutputFilePath(const TCHAR *tPath);

	// ここで渡されるhWndはIDC_EDITのHWND(コントロールのイベントだから)
	LRESULT DropInput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData);

	// ここで渡されるhWndはIDC_EDITのHWND(コントロールのイベントだから)
	LRESULT DropOutput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData);

	LRESULT TextInput(HWND hWnd, WPARAM wParam, LPARAM lParam, WNDPROC OrgSubWnd, LPVOID lpData);

	void InputRef(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void OutputRef(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void LangChange(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void OutExtChange(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void ClearOutputDir(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);

	void AppSetting(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);
};

