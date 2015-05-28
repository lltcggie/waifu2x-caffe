#pragma once

#include <windows.h>
#include <unordered_map>
#include "GUICommon.h"


// 注意
// イベントハンドラでSetWindowLongでGWL_USERDATAを書き換えた場合おかしくなる
class CControl
{
private:
	// コピー、代入の禁止
	CControl(const CControl&);
	CControl& operator =(const CControl&);

protected:
	struct stEvent
	{
		CustomEventFunc pfunc;
		LPVOID lpData;

		stEvent() : pfunc(nullptr), lpData(nullptr)
		{
		}

		stEvent(const CustomEventFunc &func, const LPVOID Data) : pfunc(func), lpData(Data)
		{
		}

		stEvent(const stEvent &st)
		{
			pfunc = st.pfunc;
			lpData = st.lpData;
		}
	};

	std::unordered_map<UINT, stEvent> mEventMap;

	HWND hSub;
	WNDPROC OrgSubWnd;
	int ResourceID;

	// ダイアログプロシージャ(実質)
	virtual LRESULT SubProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

public:
	// コンストラクタ(リソースIDを指定)
	CControl(const UINT ID);
	CControl();

	// 仮想デストラクタ(何もしない)
	virtual ~CControl();

	// 一つのメッセージにつき一つの関数しか登録できない.
	// すでにあった場合は上書きされる.
	// lpDataは登録した関数に与える好きな引数.
	// 登録できる関数は、
	// BOOL Create(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);
	// のような関数.
	// 戻り値はTRUEでもFALSEでもよい.
	void SetEventCallBack(const CustomEventFunc &func, const LPVOID lpData, const UINT uMsg);

	// カスタムコントロールを登録
	BOOL Register(LPCTSTR ClassName, const HINSTANCE hInstance);

	// ユーザーが使うのはここまで


	void RegisterFunc(HWND hWnd);

	int GetResourceID();

	// ダイアログプロシージャ(形式上)
	static LRESULT CALLBACK DispatchSubProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	// カスタムコントロールプロシージャ
	static LRESULT CALLBACK DispatchCustomProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};
