#pragma once

#include <dbghelp.h>

void CreateDump(EXCEPTION_POINTERS *pep, DWORD ThreadID, int level = 2);

LONG CALLBACK ExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo, int level);
LONG CALLBACK ExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo);
