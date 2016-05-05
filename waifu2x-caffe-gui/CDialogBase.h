#pragma once

#include <windows.h>


class CDialogBase
{
private:
	// ダイアログプロシージャ(実質) 
	virtual INT_PTR DialogProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) = 0;

protected:
	HWND hDialog;

public:
	virtual ~CDialogBase(){};

	// ダイアログを作成する 
	INT_PTR DoModal(HINSTANCE hInstance, int iDialogId, HWND hWndParent = NULL);

	HWND GetDialogHWND(void);

	// ダイアログプロシージャ(形式上)
	static INT_PTR CALLBACK DispatchDialogProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};
