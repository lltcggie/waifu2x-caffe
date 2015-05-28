#pragma once

#include <windows.h>
#include <functional>


#define SetClassFunc(FuncObject, ObjectPointer) std::bind(&FuncObject, ObjectPointer,  std::placeholders::_1,  std::placeholders::_2,  std::placeholders::_3,  std::placeholders::_4)
#define SetClassCustomFunc(FuncObject, ObjectPointer) std::bind(&FuncObject, ObjectPointer,  std::placeholders::_1,  std::placeholders::_2,  std::placeholders::_3,  std::placeholders::_4,  std::placeholders::_5)
typedef std::function<void (HWND, WPARAM, LPARAM, LPVOID)> EventFunc;
typedef std::function<LRESULT (HWND, WPARAM, LPARAM, WNDPROC, LPVOID)> CustomEventFunc;
