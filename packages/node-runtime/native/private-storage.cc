// First-party, Node-API-only bridge. No provider logic or arbitrary FFI surface.
#include <node_api.h>
#include <algorithm>
#include <cctype>
#include <cwctype>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <aclapi.h>
#include <sddl.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#endif

namespace {
struct Failure : std::runtime_error {
  std::string code;
  Failure(const char* message, const char* code = "ERR_PRIVATE_STORAGE") : std::runtime_error(message), code(code) {}
};

#ifdef _WIN32
using Handle = HANDLE;
const Handle invalid = INVALID_HANDLE_VALUE;
void closeHandle(Handle handle) { if (handle != invalid) CloseHandle(handle); }
[[noreturn]] void systemFailure(const char* message) {
  const DWORD code = GetLastError();
  throw Failure(message, code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND ? "ENOENT" :
    code == ERROR_FILE_EXISTS || code == ERROR_ALREADY_EXISTS ? "EEXIST" :
    code == ERROR_ACCESS_DENIED || code == ERROR_SHARING_VIOLATION ? "EACCES" : "ERR_PRIVATE_STORAGE");
}
std::wstring wide(const std::string& text) {
  int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0);
  if (!count && !text.empty()) throw Failure("invalid UTF-8 path", "EINVAL");
  std::wstring value(count, L'\0');
  if (count) MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), value.data(), count);
  return value;
}
std::string utf8(const std::wstring& text) {
  int count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
  if (!count && !text.empty()) throw Failure("invalid filesystem name", "EINVAL");
  std::string value(count, '\0');
  if (count) WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), value.data(), count, nullptr, nullptr);
  return value;
}
struct PrivateSecurity {
  std::vector<unsigned char> tokenData;
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  PACL acl = nullptr;
  PSID sid = nullptr;
  SECURITY_ATTRIBUTES attributes{};
  PrivateSecurity() {
    HANDLE token;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) systemFailure("cannot inspect current user");
    DWORD size = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &size);
    tokenData.resize(size);
    const BOOL success = GetTokenInformation(token, TokenUser, tokenData.data(), size, &size);
    const DWORD error = GetLastError();
    CloseHandle(token);
    if (!success) { SetLastError(error); systemFailure("cannot inspect current user"); }
    sid = reinterpret_cast<TOKEN_USER*>(tokenData.data())->User.Sid;
    LPWSTR sidText = nullptr;
    if (!ConvertSidToStringSidW(sid, &sidText)) systemFailure("cannot encode current user");
    const std::wstring sddl = std::wstring(L"D:P(A;;FA;;;") + sidText + L")";
    LocalFree(sidText);
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &descriptor, nullptr)) systemFailure("cannot create private ACL");
    BOOL present, defaulted;
    if (!GetSecurityDescriptorDacl(descriptor, &present, &acl, &defaulted) || !present || !acl) {
      LocalFree(descriptor); descriptor = nullptr;
      throw Failure("private ACL is unavailable");
    }
    attributes = { sizeof(SECURITY_ATTRIBUTES), descriptor, FALSE };
  }
  ~PrivateSecurity() { if (descriptor) LocalFree(descriptor); }
};
bool privateMode(Handle file) {
  PrivateSecurity owner;
  PACL acl = nullptr;
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  const DWORD error = GetSecurityInfo(file, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &acl, nullptr, &descriptor);
  if (error != ERROR_SUCCESS) { SetLastError(error); systemFailure("cannot inspect private ACL"); }
  SECURITY_DESCRIPTOR_CONTROL control = 0;
  DWORD revision = 0;
  bool valid = GetSecurityDescriptorControl(descriptor, &control, &revision) &&
    (control & SE_DACL_PROTECTED) && acl && IsValidAcl(acl) && acl->AceCount == 1;
  void* entry = nullptr;
  if (valid) {
    valid = GetAce(acl, 0, &entry) && static_cast<ACE_HEADER*>(entry)->AceType == ACCESS_ALLOWED_ACE_TYPE &&
      !(static_cast<ACE_HEADER*>(entry)->AceFlags & INHERITED_ACE);
    if (valid) {
      const auto* ace = static_cast<ACCESS_ALLOWED_ACE*>(entry);
      valid = EqualSid(owner.sid, const_cast<DWORD*>(&ace->SidStart)) && (ace->Mask & FILE_ALL_ACCESS) == FILE_ALL_ACCESS;
    }
  }
  LocalFree(descriptor);
  return valid;
}
void restrictFile(Handle file) {
  PrivateSecurity owner;
  const DWORD error = SetSecurityInfo(file, SE_FILE_OBJECT,
    DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, nullptr, nullptr, owner.acl, nullptr);
  if (error != ERROR_SUCCESS) { SetLastError(error); systemFailure("cannot enforce private ACL"); }
  if (!privateMode(file)) throw Failure("private ACL was not enforced");
}
BY_HANDLE_FILE_INFORMATION information(Handle file) {
  BY_HANDLE_FILE_INFORMATION value{};
  if (!GetFileInformationByHandle(file, &value)) systemFailure("cannot inspect native handle");
  return value;
}
bool same(Handle left, Handle right) {
  const auto a = information(left), b = information(right);
  return a.dwVolumeSerialNumber == b.dwVolumeSerialNumber && a.nFileIndexHigh == b.nFileIndexHigh && a.nFileIndexLow == b.nFileIndexLow;
}
#else
using Handle = int;
constexpr Handle invalid = -1;
void closeHandle(Handle handle) { if (handle != invalid) close(handle); }
[[noreturn]] void systemFailure(const char* message) {
  throw Failure(message, errno == ENOENT ? "ENOENT" : errno == EEXIST ? "EEXIST" :
    errno == EACCES || errno == EPERM ? "EACCES" : errno == ENOTDIR ? "ENOTDIR" :
    errno == ELOOP ? "ELOOP" : "ERR_PRIVATE_STORAGE");
}
struct stat information(Handle file) {
  struct stat value{};
  if (fstat(file, &value) != 0) systemFailure("cannot inspect native handle");
  return value;
}
bool same(Handle left, Handle right) {
  const auto a = information(left), b = information(right);
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}
bool privateMode(Handle file) { return (information(file).st_mode & 0777) == 0600; }
void restrictFile(Handle file) {
  if (fchmod(file, 0600) != 0) systemFailure("cannot enforce private file mode");
  if (!privateMode(file)) throw Failure("private file mode was not enforced");
}
std::string canonical(const std::string& path) {
  std::unique_ptr<char, decltype(&free)> resolved(realpath(path.c_str(), nullptr), free);
  if (!resolved) systemFailure("private storage root is unavailable");
  return resolved.get();
}
#endif

struct OwnedHandle {
  Handle value = invalid;
  explicit OwnedHandle(Handle value) : value(value) {}
  ~OwnedHandle() { closeHandle(value); }
  Handle release() { const Handle old = value; value = invalid; return old; }
};

void validateName(const std::string& name) {
  if (name.empty() || name == "." || name == ".." || name.find_first_of("/\\:\0", 0, 4) != std::string::npos ||
      name.find_first_of("<>\"|?*") != std::string::npos || name.back() == '.' || name.back() == ' ') throw Failure("unsafe private storage name", "EINVAL");
  for (unsigned char c : name) if (c < 32) throw Failure("unsafe private storage name", "EINVAL");
  std::string stem = name.substr(0, name.find('.'));
  std::transform(stem.begin(), stem.end(), stem.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (stem == "CON" || stem == "PRN" || stem == "AUX" || stem == "NUL" ||
      (stem.size() == 4 && (stem.substr(0, 3) == "COM" || stem.substr(0, 3) == "LPT") && stem[3] >= '0' && stem[3] <= '9')) throw Failure("unsafe private storage name", "EINVAL");
}
std::vector<std::string> components(const std::string& parent) {
  std::vector<std::string> result;
  if (parent.empty()) return result;
  size_t start = 0;
  for (;;) {
    const size_t end = parent.find('/', start);
    const auto part = parent.substr(start, end == std::string::npos ? end : end - start);
    validateName(part);
    result.push_back(part);
    if (end == std::string::npos) return result;
    start = end + 1;
  }
}

struct Directory {
  std::string root;
  std::string realRoot;
  std::vector<std::string> parts;
  std::vector<Handle> handles;
  bool closed = false;
#ifdef _WIN32
  std::vector<std::wstring> paths;
  std::wstring childPath(const std::string& name) const { return paths.back() + L"\\" + wide(name); }
  Handle openWindowsDirectory(const std::wstring& target, bool create) {
    Handle file = CreateFileW(target.c_str(), FILE_READ_ATTRIBUTES,
      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (file == invalid && create && (GetLastError() == ERROR_FILE_NOT_FOUND || GetLastError() == ERROR_PATH_NOT_FOUND)) {
      PrivateSecurity security;
      if (!CreateDirectoryW(target.c_str(), &security.attributes) && GetLastError() != ERROR_ALREADY_EXISTS) systemFailure("cannot create private directory");
      file = CreateFileW(target.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    }
    if (file == invalid) systemFailure("private storage directory is unavailable");
    OwnedHandle opened(file);
    const auto info = information(file);
    if (!(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) throw Failure("unsafe symlink or reparse point in private storage path");
    return opened.release();
  }
#endif
  ~Directory() { dispose(); }
  void dispose() {
    for (auto it = handles.rbegin(); it != handles.rend(); ++it) closeHandle(*it);
    handles.clear(); closed = true;
  }
  void requireOpen() const { if (closed || handles.empty()) throw Failure("native directory handle is closed"); }
  void initialize(const std::string& configuredRoot, const std::string& parent, bool create) {
    root = configuredRoot;
    if (root.empty() || root.find('\0') != std::string::npos) throw Failure("invalid private storage root path", "EINVAL");
    parts = components(parent);
#ifdef _WIN32
    std::wstring absolute = wide(root);
    std::replace(absolute.begin(), absolute.end(), L'/', L'\\');
    if (absolute.rfind(L"\\\\?\\", 0) == 0) absolute.erase(0, 4);
    if (absolute.size() < 3 || absolute[1] != L':' || absolute[2] != L'\\' || !iswalpha(absolute[0])) throw Failure("private storage requires a local absolute drive path", "EINVAL");
    DWORD flags = 0;
    const std::wstring drive = absolute.substr(0, 3);
    if (!GetVolumeInformationW(drive.c_str(), nullptr, 0, nullptr, nullptr, &flags, nullptr, 0) || !(flags & FILE_PERSISTENT_ACLS)) throw Failure("filesystem cannot enforce private ACLs");
    std::wstring current = L"\\\\?\\" + drive;
    handles.push_back(openWindowsDirectory(current, false)); paths.push_back(current);
    size_t start = 3;
    while (start < absolute.size()) {
      size_t end = absolute.find(L'\\', start);
      std::wstring part = absolute.substr(start, end == std::wstring::npos ? end : end - start);
      validateName(utf8(part));
      if (current.back() != L'\\') current += L'\\';
      current += part;
      handles.push_back(openWindowsDirectory(current, false)); paths.push_back(current);
      if (end == std::wstring::npos) break;
      start = end + 1;
    }
    for (const auto& part : parts) {
      current += L'\\'; current += wide(part);
      handles.push_back(openWindowsDirectory(current, create)); paths.push_back(current);
    }
#else
    if (root.front() != '/') throw Failure("private storage requires an absolute root path", "EINVAL");
    realRoot = canonical(root);
    Handle file = open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (file == invalid) systemFailure("unsafe private storage root");
    handles.push_back(file);
    for (const auto& part : parts) {
      file = openat(handles.back(), part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      if (file == invalid && create && errno == ENOENT) {
        if (mkdirat(handles.back(), part.c_str(), 0700) != 0 && errno != EEXIST) systemFailure("cannot create private directory");
        file = openat(handles.back(), part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      }
      if (file == invalid) {
        if (errno == ENOTDIR || errno == ELOOP) systemFailure("unsafe symlink or non-directory in private storage path");
        systemFailure("private storage parent is unavailable");
      }
      handles.push_back(file);
    }
#endif
    assertCurrent();
  }
  void assertCurrent() {
    requireOpen();
#ifdef _WIN32
    for (size_t i = 0; i < handles.size(); ++i) {
      OwnedHandle logical(openWindowsDirectory(paths[i], false));
      if (!same(handles[i], logical.value)) throw Failure("private storage namespace was replaced");
    }
#else
    if (canonical(root) != realRoot) throw Failure("private storage root was replaced");
    OwnedHandle logical(open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (logical.value == invalid) throw Failure("private storage root was replaced");
    if (!same(logical.value, handles.front())) throw Failure("private storage root was replaced");
    for (size_t i = 0; i < parts.size(); ++i) {
      OwnedHandle next(openat(logical.value, parts[i].c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
      if (next.value == invalid || !same(next.value, handles[i + 1])) throw Failure("private storage namespace was replaced");
      closeHandle(logical.value); logical.value = next.release();
    }
#endif
  }
  Handle openFile(const std::string& name, bool create) {
    requireOpen(); validateName(name);
#ifdef _WIN32
    PrivateSecurity security;
    Handle file = CreateFileW(childPath(name).c_str(), GENERIC_READ | READ_CONTROL | WRITE_DAC | (create ? GENERIC_WRITE : 0),
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, create ? &security.attributes : nullptr,
      create ? CREATE_NEW : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (file == invalid) {
      const DWORD error = GetLastError();
      if ((create && (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS)) || (!create && error == ERROR_FILE_NOT_FOUND)) return invalid;
      // CREATE_NEW reports ACCESS_DENIED for an existing directory (including
      // a junction). Confirm that collision without opening its target; other
      // access failures must still propagate instead of becoming "busy".
      if (create && error == ERROR_ACCESS_DENIED) {
        const DWORD attributes = GetFileAttributesW(childPath(name).c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY)) return invalid;
      }
      SetLastError(error);
      systemFailure("cannot open private file");
    }
    OwnedHandle opened(file);
    const auto info = information(file);
    if ((info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) || GetFileType(file) != FILE_TYPE_DISK) throw Failure("unsafe private file or reparse point");
#else
    const int flags = O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK | (create ? O_CREAT | O_EXCL | O_RDWR : O_RDONLY);
    Handle file = openat(handles.back(), name.c_str(), flags, 0600);
    if (file == invalid) {
      if ((create && errno == EEXIST) || (!create && errno == ENOENT)) return invalid;
      systemFailure("cannot open unsafe or unavailable private file");
    }
    OwnedHandle opened(file);
    if (!S_ISREG(information(file).st_mode)) throw Failure("unsafe private file type");
#endif
    if (create) restrictFile(file);
    return opened.release();
  }
  bool remove(const std::string& name) {
    requireOpen(); validateName(name);
#ifdef _WIN32
    if (DeleteFileW(childPath(name).c_str())) return true;
    if (GetLastError() == ERROR_FILE_NOT_FOUND) return false;
#else
    if (unlinkat(handles.back(), name.c_str(), 0) == 0) return true;
    if (errno == ENOENT) return false;
#endif
    systemFailure("cannot remove private file");
  }
  bool link(const std::string& from, const std::string& to) {
    requireOpen(); validateName(from); validateName(to);
#ifdef _WIN32
    if (CreateHardLinkW(childPath(to).c_str(), childPath(from).c_str(), nullptr)) return true;
    if (GetLastError() == ERROR_ALREADY_EXISTS || GetLastError() == ERROR_FILE_EXISTS) return false;
#else
    if (linkat(handles.back(), from.c_str(), handles.back(), to.c_str(), 0) == 0) return true;
    if (errno == EEXIST) return false;
#endif
    systemFailure("cannot link private file");
  }
  void rename(const std::string& from, const std::string& to) {
    requireOpen(); validateName(from); validateName(to);
#ifdef _WIN32
    if (!MoveFileExW(childPath(from).c_str(), childPath(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) systemFailure("cannot atomically replace private file");
#else
    if (renameat(handles.back(), from.c_str(), handles.back(), to.c_str()) != 0) systemFailure("cannot atomically replace private file");
#endif
  }
  void sync() {
    requireOpen();
#ifndef _WIN32
    if (fsync(handles.back()) != 0) systemFailure("cannot flush private directory");
#endif
    // Windows commits file data with FlushFileBuffers and rename with WRITE_THROUGH.
  }
  std::vector<std::string> list() {
    requireOpen();
    std::vector<std::string> names;
#ifdef _WIN32
    WIN32_FIND_DATAW entry{};
    HANDLE search = FindFirstFileW((paths.back() + L"\\*").c_str(), &entry);
    if (search == INVALID_HANDLE_VALUE) {
      if (GetLastError() == ERROR_FILE_NOT_FOUND) return names;
      systemFailure("cannot list private directory");
    }
    do {
      const std::wstring name(entry.cFileName);
      if (name != L"." && name != L"..") names.push_back(utf8(name));
    } while (FindNextFileW(search, &entry));
    const DWORD error = GetLastError(); FindClose(search);
    if (error != ERROR_NO_MORE_FILES) { SetLastError(error); systemFailure("cannot list private directory"); }
#else
    // A new open description avoids sharing a directory stream's offset.
    OwnedHandle copy(openat(handles.back(), ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (copy.value == invalid) systemFailure("cannot list private directory");
    DIR* directory = fdopendir(copy.value);
    if (!directory) systemFailure("cannot list private directory");
    copy.release();
    errno = 0;
    while (const auto* entry = readdir(directory)) {
      const std::string name(entry->d_name);
      if (name != "." && name != "..") names.push_back(name);
      errno = 0;
    }
    const int error = errno; closedir(directory);
    if (error) { errno = error; systemFailure("cannot list private directory"); }
#endif
    return names;
  }
};

struct File {
  Handle handle;
  bool writable;
  File(Handle handle, bool writable) : handle(handle), writable(writable) {}
  ~File() { dispose(); }
  void dispose() { closeHandle(handle); handle = invalid; }
  void requireOpen() { if (handle == invalid) throw Failure("native file handle is closed"); }
  void write(const std::string& text) {
    requireOpen();
    if (!writable) throw Failure("native file handle is read-only");
    restrictFile(handle);
    size_t offset = 0;
#ifdef _WIN32
    LARGE_INTEGER zero{};
    if (!SetFilePointerEx(handle, zero, nullptr, FILE_BEGIN)) systemFailure("cannot seek private file");
    while (offset < text.size()) {
      DWORD written = 0;
      const DWORD count = static_cast<DWORD>(std::min<size_t>(text.size() - offset, 1024 * 1024));
      if (!WriteFile(handle, text.data() + offset, count, &written, nullptr) || !written) systemFailure("cannot write private file");
      offset += written;
    }
    if (!SetEndOfFile(handle) || !FlushFileBuffers(handle)) systemFailure("cannot flush private file");
#else
    while (offset < text.size()) {
      const auto written = pwrite(handle, text.data() + offset, text.size() - offset, static_cast<off_t>(offset));
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) systemFailure("cannot write private file");
      offset += static_cast<size_t>(written);
    }
    if (ftruncate(handle, static_cast<off_t>(text.size())) != 0 || fsync(handle) != 0) systemFailure("cannot flush private file");
#endif
  }
};

const napi_type_tag directoryTag = { 0x42fe954c28f17211ULL, 0x9d0008269560d8f1ULL };
const napi_type_tag fileTag = { 0x42fe954c28f17212ULL, 0x9d0008269560d8f2ULL };
void check(napi_status status) { if (status != napi_ok) throw Failure("invalid native API argument or receiver"); }
napi_value undefined(napi_env env) { napi_value value; check(napi_get_undefined(env, &value)); return value; }
napi_value boolean(napi_env env, bool flag) { napi_value value; check(napi_get_boolean(env, flag, &value)); return value; }
std::string string(napi_env env, napi_value value) {
  size_t size = 0; check(napi_get_value_string_utf8(env, value, nullptr, 0, &size));
  std::vector<char> bytes(size + 1); check(napi_get_value_string_utf8(env, value, bytes.data(), bytes.size(), &size));
  return std::string(bytes.data(), size);
}
struct Call {
  napi_value receiver;
  napi_value args[3]{};
  size_t count = 3;
  Call(napi_env env, napi_callback_info info) { check(napi_get_cb_info(env, info, &count, args, &receiver, nullptr)); }
  void require(size_t size) { if (count < size) throw Failure("missing native API argument"); }
};
template<class T> T* receiver(napi_env env, napi_value value, const napi_type_tag* tag) {
  bool valid = false; check(napi_check_object_type_tag(env, value, tag, &valid));
  if (!valid) throw Failure("invalid native handle receiver");
  void* data = nullptr; check(napi_unwrap(env, value, &data));
  if (!data) throw Failure("native handle is unavailable");
  return static_cast<T*>(data);
}
template<class F> napi_value invoke(napi_env env, F callback) {
  try { return callback(); }
  catch (const Failure& error) { napi_throw_error(env, error.code.c_str(), error.what()); }
  catch (const std::exception&) { napi_throw_error(env, "ERR_PRIVATE_STORAGE", "native storage operation failed"); }
  return nullptr;
}

napi_value fileClose(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); receiver<File>(env, call.receiver, &fileTag)->dispose(); return undefined(env); }); }
napi_value fileRestrict(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); auto file = receiver<File>(env, call.receiver, &fileTag); file->requireOpen(); restrictFile(file->handle); return undefined(env); }); }
napi_value filePrivate(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); auto file = receiver<File>(env, call.receiver, &fileTag); file->requireOpen(); return boolean(env, privateMode(file->handle)); }); }
napi_value fileWrite(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); call.require(1); receiver<File>(env, call.receiver, &fileTag)->write(string(env, call.args[0])); return undefined(env); }); }

napi_value wrapFile(napi_env env, Handle handle, bool writable) {
  if (handle == invalid) { napi_value value; check(napi_get_null(env, &value)); return value; }
  auto file = std::make_unique<File>(handle, writable);
  napi_value value; check(napi_create_object(env, &value));
  check(napi_type_tag_object(env, value, &fileTag));
  napi_property_descriptor properties[] = {
    {"close", nullptr, fileClose, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"restrict", nullptr, fileRestrict, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"isPrivate", nullptr, filePrivate, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"writeText", nullptr, fileWrite, nullptr, nullptr, nullptr, napi_default, nullptr}
  };
  check(napi_define_properties(env, value, 4, properties));
  check(napi_wrap(env, value, file.get(), [](napi_env, void* data, void*) { delete static_cast<File*>(data); }, nullptr, nullptr));
  file.release(); return value;
}
napi_value directoryClose(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); receiver<Directory>(env, call.receiver, &directoryTag)->dispose(); return undefined(env); }); }
napi_value directoryAssert(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); receiver<Directory>(env, call.receiver, &directoryTag)->assertCurrent(); return undefined(env); }); }
napi_value directorySync(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); receiver<Directory>(env, call.receiver, &directoryTag)->sync(); return undefined(env); }); }
napi_value directoryCreate(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); call.require(1); return wrapFile(env, receiver<Directory>(env, call.receiver, &directoryTag)->openFile(string(env, call.args[0]), true), true); }); }
napi_value directoryOpen(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); call.require(1); return wrapFile(env, receiver<Directory>(env, call.receiver, &directoryTag)->openFile(string(env, call.args[0]), false), false); }); }
napi_value directoryRemove(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); call.require(1); return boolean(env, receiver<Directory>(env, call.receiver, &directoryTag)->remove(string(env, call.args[0]))); }); }
napi_value directoryRename(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); call.require(2); receiver<Directory>(env, call.receiver, &directoryTag)->rename(string(env, call.args[0]), string(env, call.args[1])); return undefined(env); }); }
napi_value directoryLink(napi_env env, napi_callback_info info) { return invoke(env, [&] { Call call(env, info); call.require(2); return boolean(env, receiver<Directory>(env, call.receiver, &directoryTag)->link(string(env, call.args[0]), string(env, call.args[1]))); }); }
napi_value directoryList(napi_env env, napi_callback_info info) { return invoke(env, [&] {
  Call call(env, info); const auto names = receiver<Directory>(env, call.receiver, &directoryTag)->list();
  napi_value value; check(napi_create_array_with_length(env, names.size(), &value));
  for (size_t i = 0; i < names.size(); ++i) { napi_value name; check(napi_create_string_utf8(env, names[i].data(), names[i].size(), &name)); check(napi_set_element(env, value, static_cast<uint32_t>(i), name)); }
  return value;
}); }
napi_value openDirectory(napi_env env, napi_callback_info info) { return invoke(env, [&] {
  Call call(env, info); call.require(3); bool create; check(napi_get_value_bool(env, call.args[2], &create));
  auto directory = std::make_unique<Directory>();
  directory->initialize(string(env, call.args[0]), string(env, call.args[1]), create);
  napi_value value; check(napi_create_object(env, &value)); check(napi_type_tag_object(env, value, &directoryTag));
  napi_property_descriptor properties[] = {
    {"close", nullptr, directoryClose, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"assertCurrent", nullptr, directoryAssert, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"sync", nullptr, directorySync, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"createFile", nullptr, directoryCreate, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"openFile", nullptr, directoryOpen, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"remove", nullptr, directoryRemove, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"rename", nullptr, directoryRename, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"link", nullptr, directoryLink, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"list", nullptr, directoryList, nullptr, nullptr, nullptr, napi_default, nullptr}
  };
  check(napi_define_properties(env, value, 9, properties));
  check(napi_wrap(env, value, directory.get(), [](napi_env, void* data, void*) { delete static_cast<Directory*>(data); }, nullptr, nullptr));
  directory.release(); return value;
}); }
} // namespace

NAPI_MODULE_INIT() {
  napi_property_descriptor property = {"openDirectory", nullptr, openDirectory, nullptr, nullptr, nullptr, napi_default, nullptr};
  if (napi_define_properties(env, exports, 1, &property) != napi_ok) return nullptr;
  napi_value version;
  if (napi_create_uint32(env, 1, &version) != napi_ok || napi_set_named_property(env, exports, "version", version) != napi_ok) return nullptr;
  return exports;
}
