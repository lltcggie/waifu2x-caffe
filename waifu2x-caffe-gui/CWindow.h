#pragma once

#include <windows.h>
#include <vector>
#include <unordered_map>
#include "CWindowBase.h"
#include "GUICommon.h"


// 注意
// イベントハンドラでSetWindowLongでGWL_USERDATAを書き換えた場合おかしくなる
class CWindow: public CWindowBase
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
	std::unordered_map<UINT, stEvent> mEvent;

	// コピー、代入の禁止
	CWindow(const CWindow&);
	CWindow& operator =(const CWindow&);

	// ダイアログプロシージャ（実質）
	LRESULT WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

public:
	// 一つのメッセージにつき一つの関数しか登録できない.
	// すでにあった場合は上書きされる.
	// lpDataは登録した関数に与える好きな引数.
	// 登録できる関数は、
	// BOOL Create(HWND hWnd, WPARAM wParam, LPARAM lParam, LPVOID lpData);
	// のような関数.
	// 戻り値はTRUEでもFALSEでもよい.
	void SetEventCallBack(EventFunc pfunc, LPVOID lpData, UINT uMsg);

	// ウィンドウサイズ変更
	void SetWindowSize(int nWidth, int nHeight, BOOL Adjust);

	// ウィンドウを画面中心へ移動
	void MoveWindowCenter();

	// コンストラクタ(何もしない)
	CWindow();
};
