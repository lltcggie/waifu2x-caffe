#include <windows.h>
#include "CDialogBase.h"


// ダイアログを作成する
INT_PTR CDialogBase::DoModal(HINSTANCE hInstance, int iDialogId, HWND hWndParent)
{ 
	return DialogBoxParam(hInstance, MAKEINTRESOURCE(iDialogId), hWndParent, &DispatchDialogProc, (LPARAM)this);
}

HWND CDialogBase::GetDialogHWND(void)
{
	return hDialog;
}

// ダイアログプロシージャ(形式上) 
INT_PTR CALLBACK CDialogBase::DispatchDialogProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	// ダイアログの 32 ビット整数に格納されている  
	// this ポインタを取りだす
	CDialogBase *pcDialog = (CDialogBase *)GetWindowLongPtr(hWnd, GWLP_USERDATA); 
	if(pcDialog == NULL) 
	{
		if(uMsg == WM_INITDIALOG || uMsg == WM_CREATE) 
		{ 
			// 直前に DialogBoxParam() が呼ばれてる場合
			// this ポインタをダイアログのユーザー領域に入れる
			pcDialog = (CDialogBase*)lParam;

			SetWindowLongPtr(hWnd, GWLP_USERDATA, (LONG_PTR)pcDialog);
			pcDialog->hDialog = hWnd;

			return pcDialog->DialogProc(hWnd, uMsg, wParam, lParam);
		}

		return FALSE; 
	}

	// メンバ関数のダイアログプロシージャを呼び出す 
	return pcDialog->DialogProc(hWnd, uMsg, wParam, lParam);
} 
