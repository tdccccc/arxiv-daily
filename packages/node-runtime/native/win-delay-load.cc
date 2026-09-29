// Bind Node-API imports to the hosting Node/Electron executable, not a new node.exe.
#include <windows.h>
#include <delayimp.h>
#include <cstring>

static FARPROC loadNode(unsigned notification, PDelayLoadInfo info) {
  if (notification != dliNotePreLoadLibrary || _stricmp(info->szDll, "node.exe") != 0) return nullptr;
  HMODULE host = GetModuleHandleW(nullptr);
  if (!host || !GetProcAddress(host, "napi_get_last_error_info")) host = GetModuleHandleW(L"node.dll");
  return reinterpret_cast<FARPROC>(host);
}

extern "C" { PfnDliHook __pfnDliNotifyHook2 = loadNode; }
