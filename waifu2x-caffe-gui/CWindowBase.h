#pragma once

#include <windows.h>


class CWindowBase
{
private:
	// ダイアログプロシージャ(実質)
	virtual LRESULT WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) = 0;

protected:
	HWND hWindow;
	DWORD dwStyle;

public:
	// 仮想デストラクタ(何もしない)
	virtual ~CWindowBase();

	// ダイアログを作成
	// Adjust: 真ならサイズをクライアント領域のものとする
	// bSizeBox: 真ならサイズ変更できるようにする
	HWND InitWindow(HINSTANCE hInstance, UINT Width, UINT Height,
		BOOL Adjust, BOOL bSizeBox, LPCTSTR szClassName, LPCTSTR szWindowTitle);

	HWND InitWindow(HINSTANCE hInstance, UINT Width, UINT Height,
		BOOL Adjust, LPCTSTR szClassName, LPCTSTR szWindowTitle,
		UINT WindowClassStyle = CS_HREDRAW | CS_VREDRAW, DWORD WindowStyle = WS_OVERLAPPEDWINDOW);

	// ウィンドウを表示
	void ShowWindow(int nCmdShow);

	// メインウィンドウのハンドルを取得
	HWND GetWindowHandle(void);

	// メッセージループ
	void MessageLoop();

	// 戻り値:	終了 0
	//			メッセージを処理 1
	//			メッセージはなかった 2
	int PeekLoop();


	// ダイアログプロシージャ(形式上)
	static LRESULT CALLBACK DispatchWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};
