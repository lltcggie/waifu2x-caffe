#include <windows.h>
#include <tchar.h>
#include <dbghelp.h>
#include <stdio.h>
#include <crtdbg.h>
#include <shlwapi.h>
#include "MiniDump.h"

// #pragma comment (lib, "dbghelp.lib")
#pragma comment (lib, "shlwapi.lib")

typedef BOOL (WINAPI *MINIDUMPWRITEDUMP)(HANDLE hProcess, DWORD ProcessId, HANDLE hFile, MINIDUMP_TYPE DumpType, PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam, PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam, PMINIDUMP_CALLBACK_INFORMATION CallbackParam);

const int DefaultLevel = 2;


void CreateDump(EXCEPTION_POINTERS *pep, DWORD ThreadID, int level)
{
	HMODULE mhLib = ::LoadLibrary(_T("dbghelp_local.dll"));
	if (!mhLib)
		return;

	MINIDUMPWRITEDUMP pDump = (MINIDUMPWRITEDUMP)::GetProcAddress(mhLib, "MiniDumpWriteDump");
	if (!pDump)
	{
		FreeLibrary(mhLib);

		if (pep)
			UnhandledExceptionFilter(pep);

		return;
	}

	TCHAR szFilePath[1024];

	::GetModuleFileName(NULL, szFilePath, sizeof(szFilePath));
	::PathRemoveFileSpec(szFilePath);
	_tcscat_s(szFilePath, TEXT("\\"));

	TCHAR SettingDir[100] = TEXT("");

#if !defined(PRODUCT_MODE)
	{
		TCHAR iniPath[1024];
		_tcscpy_s(iniPath, szFilePath);
		_tcscat_s(iniPath, TEXT("DebugSetting.ini"));
		GetPrivateProfileString(TEXT("MemoryDump"), TEXT("RelativePath"), TEXT(""), SettingDir, sizeof(SettingDir) / sizeof(SettingDir[0]), iniPath);

		const size_t len = _tcslen(SettingDir);
		if (len > 0 && (SettingDir[len - 1] != TEXT('\\') || SettingDir[len - 1] != TEXT('/')))
			_tcscat_s(SettingDir, TEXT("\\"));
	}
#endif

	SYSTEMTIME systime;
	GetLocalTime(&systime);

	TCHAR buf[100];
	_stprintf_s(buf, TEXT("%04d_%02d_%02d %02u_%02u_%02u_%03u.dmp"), systime.wYear, systime.wMonth, systime.wDay, systime.wHour, systime.wMinute, systime.wSecond, systime.wMilliseconds);

	
	_tcscat_s(szFilePath, SettingDir);
	_tcscat_s(szFilePath, buf);

	HANDLE hFile = CreateFile(szFilePath,
		GENERIC_READ | GENERIC_WRITE,
		0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE){
		_tprintf(_T("CreateFile failed. Error: %u \n"),
			GetLastError());
		FreeLibrary(mhLib);

		if (pep)
			UnhandledExceptionFilter(pep);

		return;
	}

	MINIDUMP_EXCEPTION_INFORMATION mdei;

	mdei.ThreadId = ThreadID;
	mdei.ExceptionPointers = pep;
	mdei.ClientPointers = FALSE;

	MINIDUMP_CALLBACK_INFORMATION mci;

	mci.CallbackRoutine = NULL;
	mci.CallbackParam = 0;

	MINIDUMP_TYPE mdt;

	switch (level)
	{
	case 0:
		mdt = (MINIDUMP_TYPE)(MiniDumpNormal);
		break;
	case 1:
		mdt = (MINIDUMP_TYPE)(
			MiniDumpWithIndirectlyReferencedMemory |
			MiniDumpScanMemory);
		break;
	case 2:
		mdt = (MINIDUMP_TYPE)(
			MiniDumpWithIndirectlyReferencedMemory |
			MiniDumpScanMemory |
			MiniDumpWithDataSegs |
			MiniDumpWithHandleData |
			MiniDumpWithFullMemoryInfo |
			MiniDumpWithThreadInfo |
			MiniDumpWithUnloadedModules);
		break;
	case 3:
		mdt = (MINIDUMP_TYPE)(
			MiniDumpWithPrivateReadWriteMemory |
			MiniDumpWithDataSegs |
			MiniDumpWithHandleData |
			MiniDumpWithFullMemoryInfo |
			MiniDumpWithThreadInfo |
			MiniDumpWithUnloadedModules);
		break;
	default:
		mdt = (MINIDUMP_TYPE)(
			MiniDumpWithFullMemory |
			MiniDumpWithFullMemoryInfo |
			MiniDumpWithHandleData |
			MiniDumpWithThreadInfo |
			MiniDumpWithUnloadedModules);
		break;
	}

	BOOL rv = pDump(
		GetCurrentProcess(), GetCurrentProcessId(),
		hFile, mdt, (pep != NULL) ? &mdei : NULL, NULL, &mci);
	if (rv == FALSE){
		// _tprintf(_T("MiniDumpWriteDump failed. Error: %u \n"), GetLastError());
	}

	CloseHandle(hFile);

	FreeLibrary(mhLib);

	return;
}


struct stException
{
	PEXCEPTION_POINTERS ExceptionInfo;
	DWORD ThreadID;
	int Level;
};

static stException g_info;


DWORD WINAPI ReportFunc(LPVOID ThreadParam)
{
	CreateDump(g_info.ExceptionInfo, g_info.ThreadID, g_info.Level);
	return 0;
}

LONG CALLBACK ExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo, int level)
{
	if (!ExceptionInfo || ExceptionInfo->ExceptionRecord->ExceptionCode != 0xE06D7363)
	{
		g_info.ExceptionInfo = ExceptionInfo;
		g_info.ThreadID = GetCurrentThreadId();
		g_info.Level = level;
		HANDLE hThread = CreateThread(NULL, 0, ReportFunc, NULL, 0, NULL);
		WaitForSingleObject(hThread, INFINITE);
	}

	return EXCEPTION_CONTINUE_SEARCH;
}

LONG CALLBACK ExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo)
{
	return ExceptionHandler(ExceptionInfo, DefaultLevel);
}
