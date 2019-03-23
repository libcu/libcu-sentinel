/*
sentinel_int.h - internal include
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _SENTINEL_INT_H
#define _SENTINEL_INT_H

//////////////////////
// OS
#pragma region OS

/* Figure out if we are dealing with Unix, Windows, or some other operating system. */
#if defined(__OS_OTHER)
# if __OS_OTHER == 1
#  undef __OS_UNIX
#  define __OS_UNIX 0
#  undef __OS_WIN
#  define __OS_WIN 0
# else
#  undef __OS_OTHER
# endif
#endif
#if !defined(__OS_UNIX) && !defined(__OS_OTHER)
# define __OS_OTHER 0
# ifndef __OS_WIN
#  if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#   define __OS_WIN 1
#   define __OS_UNIX 0
#  else
#   define __OS_WIN 0
#   define __OS_UNIX 1
#  endif
# else
#  define __OS_UNIX 0
# endif
#else
# ifndef __OS_WIN
#  define __OS_WIN 0
# endif
#endif

#if __OS_WIN
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <crtdefs.h>
//#define _uintptr_t uintptr_t
#elif __OS_UNIX
#endif

#ifdef __CUDA_ARCH__
#if __OS_WIN
#define panic(fmt, ...) { printf(fmt"\n", __VA_ARGS__); asm("trap;"); }
#elif __OS_UNIX
#define panic(fmt, ...) { printf(fmt"\n"); asm("trap;"); }
#endif
#else
//__forceinline__ void Coverage(int line) { }
#if __OS_WIN
#define panic(fmt, ...) { printf(fmt"\n", __VA_ARGS__); exit(1); }
#elif __OS_UNIX
#define panic(fmt, ...) { printf(fmt"\n"); exit(1); }
#endif
#endif /* __CUDA_ARCH__ */

#pragma endregion

//////////////////////
// DEVICE/HOST
#pragma region DEVICE/HOST
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifndef __CUDA_ARCH__
#define __host_device__ __host__
#define __hostb_device__
#define __host_constant__
#else
#define __host_device__ __device__
#define __hostb_device__ __device__
#define __host_constant__ __constant__
#endif
#ifdef __cplusplus
}
#endif

#pragma endregion

//////////////////////
// RUNTIME
#pragma region RUNTIME
#ifdef __cplusplus
extern "C" {;
#endif

/* PTX conditionals */
#ifdef _WIN64
#define _UX ".u64"
#define _BX ".b64"
#define __R "l"
#define __I "r"
#else
#define _UX ".u32"
#define _BX ".b32"
#define __R "r"
#define __I "r"
#endif

/* Memory allocation - rounds up to 8 */
#define ROUND8_(x)			(((x)+7)&~7)
/* Returns the length of an array at compile time (via math) */
#define ARRAYSIZE_(symbol) (sizeof(symbol) / sizeof(symbol[0]))

#ifdef __CUDA_ARCH__

/* Compare S1 and S2.  */
extern __device__ int strcmp_(const char *s1, const char *s2);
#define strcmp strcmp_

/* Return the length of S.  */
extern __device__ size_t strlen_(const char *s);
#define strlen strlen_

#endif

extern bool gpuAssert(cudaError_t code, const char *action, const char *file = nullptr, int line = 0, bool abort = true);
//extern char **cudaDeviceTransferStringArray(size_t length, char *const value[], cudaError_t *error = nullptr);

#ifdef __cplusplus
};
#endif

#define cudaErrorCheck(x) { gpuAssert((x), #x, __FILE__, __LINE__, true); }
#define cudaErrorCheckA(x) { gpuAssert((x), #x, __FILE__, __LINE__, false); }
#define cudaErrorCheckF(x, f) { if (!gpuAssert((x), #x, __FILE__, __LINE__, false)) f; }
#define cudaErrorCheckLast() { gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }

#pragma endregion

#endif  /* _SENTINEL_INT_H */