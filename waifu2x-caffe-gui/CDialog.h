#pragma once

#include <windows.h>
#include <vector>
#include <unordered_map>
#include "CDialogBase.h"
#include "GUICommon.h"

class CControl;


// 注意
// イベントハンドラでSetWindowLongでGWL_USERDATAを書き換えた場合おかしくなる
class CDialog: public CDialogBase
{
private:
	struct stEvent
	{
		EventFunc pfunc;
		LPVOID lpData;

		stEvent() : pfunc(nullptr), lpData(nullptr)
		{
		}

		stEvent(const EventFunc &func, const LPVOID Data) : pfunc(func), lpData(Data)
		{
		}

		stEvent(const stEvent &st)
		{
			pfunc = st.pfunc;
			lpData = st.lpData;
		}
	};

	struct stCommand
	{
		EventFunc pfunc;
		LPVOID lpData;

		stCommand() : pfunc(nullptr), lpData(nullptr)
		{
		}

		stCommand(const EventFunc &func, const LPVOID Data) : pfunc(func), lpData(Data)
		{
		}

		stCommand(const stCommand &st)
		{
			pfunc = st.pfunc;
			lpData = st.lpData;
		}
	};

	std::vector<CControl*> vControl;
	std::unordered_map<UINT, stEvent> mEventMap;
	std::unordered_map<UINT, stCommand> mCommandMap;

	EventFunc mInitFunc;
	LPVOID mInitData;

	// コピー、代入の禁止
	CDialog(const CDialog&);
	CDialog& operator =(const CDialog&);

	// ダイアログプロシージャ
	INT_PTR DialogProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	void CommandCallBack(HWND hWnd, WPARAM wParam, LPARAM lParam);
	void SetControl(HWND hWnd);

public:
	// 一つのメッセージにつき一つの関数しか登録できない.
	// すでにあった場合は上書きされる.
	// WM_COMMANDを登録した場合、SetCommandCallBackは使えなくなる.
	// lpDataは登録した関数に与える好きな引数.
	// 登録できる関数は、
	// BOOL Create(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);
	// のような関数.
	// ダイアログではTRUEを返すこと.
	void SetEventCallBack(const EventFunc &func, const LPVOID lpData, const UINT uMsg);

	// ボタンが押されたときなどのため
	// lpDataは登録した関数に与える好きな引数
	void SetCommandCallBack(const EventFunc &func, const LPVOID lpData, const UINT ResourceID);

	// ボタンなどのサブクラス化するコントロールを追加する
	void AddControl(CControl *pfunc);

	// コンストラクタ(何もしない)
	CDialog();
};
