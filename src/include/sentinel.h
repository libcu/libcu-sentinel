/*
sentinel.h - lite message bus framework for device to host functions
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

#ifndef _SENTINEL_H
#define _SENTINEL_H

#include <cuda_runtime.h>

//////////////////////
// RUNTIME
#pragma region RUNTIME
#ifdef __cplusplus
extern "C" {;
#endif

/* OS */

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

/* DEVICE/HOST */
#ifndef __CUDA_ARCH__
#define __host_device__ __host__
#define __hostb_device__
#define __host_constant__
#else
#define __host_device__ __device__
#define __hostb_device__ __device__
#define __host_constant__ __constant__
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

#define cudaErrorCheck(x) { gpuAssert((x), #x, __FILE__, __LINE__, true); }
#define cudaErrorCheckA(x) { gpuAssert((x), #x, __FILE__, __LINE__, false); }
#define cudaErrorCheckF(x, f) { if (!gpuAssert((x), #x, __FILE__, __LINE__, false)) f; }
#define cudaErrorCheckLast() { gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }

#ifdef __cplusplus
};
#endif
#pragma endregion

//////////////////////
// PIPELINE
#pragma region PIPELINE
#ifdef __cplusplus
extern "C" {;
#endif

#include <stdio.h>
#ifdef _MSC_VER
#ifndef STRICT
#define STRICT
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
/* OS type of fd. */
typedef HANDLE FDTYPE;
/* OS type of pid. */
typedef HANDLE PIDTYPE;
/* OS bad fd. */
#define __BAD_FD INVALID_HANDLE_VALUE
/*  OS bad pid. */
#define __BAD_PID INVALID_HANDLE_VALUE
#else
/* OS type of fd. */
typedef int FDTYPE;
/* OS type of pid. */
typedef int PIDTYPE;
/* OS bad fd. */
#define __BAD_FD -1
/* OS bad pid. */
#define __BAD_PID -1
#endif

struct pipelineRedir {
	FILE *in;
	FILE *out;
	FILE *err;
	FDTYPE input;
	FDTYPE output;
	FDTYPE error;
};

/* Cleanup the pipeline's children. */
extern int pipelineCleanup(int numPids, PIDTYPE *pids, int child_siginfo);
/* Creates the pipeline. */
extern int pipelineCreate(int argc, char **argv, PIDTYPE **pidsPtr, FDTYPE *inPipePtr, FDTYPE *outPipePtr, FDTYPE *errFilePtr, FDTYPE process, pipelineRedir *redirs);
/* Ack the pipeline. */
extern void pipelineOpen(pipelineRedir &redir);
/* Ack the pipeline. */
extern void pipelineClose(pipelineRedir &redir);
/* Reads from the pipeline. */
extern void pipelineRead(pipelineRedir &redir);

#ifdef __cplusplus
};
#endif
#pragma endregion


//////////////////////
// MUTEX
#pragma region MUTEX
#ifdef __cplusplus
extern "C" {;
#endif

#define MUTEXPRED_EQ 1
#define MUTEXPRED_NE 2
#define MUTEXPRED_LT 3
#define MUTEXPRED_GT 4
#define MUTEXPRED_LTE 5
#define MUTEXPRED_GTE 6
#define MUTEXPRED_AND 7
#define MUTEXPRED_ANE 8

typedef struct mutexSleep_t {
	float msmin;
	float msmax;
	float factor;
	float ms;
} mutexSleep_t;

/* Mutex default sleep */
extern __hostb_device__ mutexSleep_t mutexDefaultSleep;

/* Mutex with exponential back-off. */
extern __host_device__ void mutexSpinLock(void **cancelToken, volatile long *mutex, long cmp = 0, long val = 1, char pred = 0, long predVal = 0, bool(*func)(void **) = nullptr, void **funcTag = nullptr, mutexSleep_t *ms = nullptr);

/* Mutex set. */
extern __host_device__ void mutexSet(volatile long *mutex, long val = 0);

/* Mutex held. */
#define mutexHeld(mutex) (*mutex == 1)

#ifdef __cplusplus
};
#endif
#pragma endregion

//////////////////////
// SENTINEL
#pragma region SENTINEL

#include <driver_types.h>
#include <stdio.h> //? remove
#if __OS_WIN
#include <fcntl.h>
#include <io.h>
#endif
#ifdef __cplusplus
extern "C" {;
#endif

#define HAS_HOSTSENTINEL 0 //: FLIP THIS

#ifndef HAS_DEVICESENTINEL
#define HAS_DEVICESENTINEL 1
#endif
#ifndef HAS_HOSTSENTINEL
#define HAS_HOSTSENTINEL 1
#endif

#define SENTINEL_NAME "Sentinel" //"Global\\Sentinel"
#define SENTINEL_MAGIC (unsigned short)0xC811
#define SENTINEL_DEVICEMAPS 1
#define SENTINEL_MSGSIZE 5120
#define SENTINEL_MSGCOUNT 1
#define SENTINEL_CHUNK 4096

typedef struct sentinelInPtr {
	void *field;
	int size;
	void *unknown;
} sentinelInPtr;

typedef struct sentinelOutPtr {
	void *field;
	void *buf;
	int size;
	void *sizeField;
	void *unknown;
} sentinelOutPtr;

#define SENTINELFLOW_NONE 0
#define SENTINELFLOW_WAIT 1
#define SENTINELFLOW_TRAN 2

typedef struct sentinelMessage {
	unsigned short op;
	unsigned char flow;
	int size;
	char *(*prepare)(void*, char*, char*, intptr_t);
	bool(*postfix)(void*, intptr_t);
	__device__ sentinelMessage(unsigned short op, unsigned char flow = SENTINELFLOW_WAIT, int size = 0, char *(*prepare)(void*, char*, char*, intptr_t) = nullptr, bool(*postfix)(void*, intptr_t) = nullptr)
		: op(op), flow(flow), size(size), prepare(prepare), postfix(postfix) { }
} sentinelMessage;
#define SENTINELPREPARE(P) ((char *(*)(void*,char*,char*,intptr_t))&P)
#define SENTINELPOSTFIX(P) ((bool (*)(void*,intptr_t))&P)

typedef struct sentinelClientMessage {
	sentinelMessage base;
	pipelineRedir redir;
	sentinelClientMessage(pipelineRedir redir, unsigned short op, unsigned char flow = SENTINELFLOW_WAIT, int size = 0, char *(*prepare)(void*, char*, char*, intptr_t) = nullptr, bool(*postfix)(void*, intptr_t) = nullptr)
		: base(op, flow, size, prepare, postfix), redir(redir) { }
} sentinelClientMessage;

typedef struct __align__(8) {
	unsigned short magic;
	volatile long control;
	int locks;
	int length;
	char data[SENTINEL_MSGSIZE];
	void dump();
} sentinelCommand;

typedef struct __align__(8) {
	long getId;
	volatile long setId;
	intptr_t offset;
	sentinelCommand cmds[SENTINEL_MSGCOUNT];
	void dump();
} sentinelMap;

typedef struct sentinelExecutor {
	sentinelExecutor *next;
	const char *name;
	bool(*executor)(void*, sentinelMessage*, int, char*(**)(void*, char*, char*, intptr_t));
	void *tag;
} sentinelExecutor;

typedef struct sentinelContext {
	sentinelMap *deviceMap[SENTINEL_DEVICEMAPS];
	sentinelMap *hostMap;
	sentinelExecutor *hostList;
	sentinelExecutor *deviceList;
} sentinelContext;

//#if HAS_HOSTSENTINEL // not-required
//	extern sentinelMap *_sentinelHostMap;
//	extern intptr_t _sentinelHostMapOffset;
//#endif
#if HAS_DEVICESENTINEL
extern __constant__ const sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
#endif

extern void sentinelServerInitialize(sentinelExecutor *deviceExecutor = nullptr, char *mapHostName = (char *)SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);
extern void sentinelServerShutdown();
#if HAS_DEVICESENTINEL
extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn = nullptr, sentinelOutPtr *ptrsOut = nullptr);
#endif
#if HAS_HOSTSENTINEL
extern void sentinelClientInitialize(char *mapHostName = (char *)SENTINEL_NAME);
extern void sentinelClientShutdown();
extern void sentinelClientRedir(pipelineRedir *redir);
extern void sentinelClientSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn = nullptr, sentinelOutPtr *ptrsOut = nullptr);
#endif
extern sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);
extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);
extern void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);

#define SENTINELCONTROL_NORMAL 0x0
#define SENTINELCONTROL_DEVICE 0x1
#define SENTINELCONTROL_DEVICERDY 0x2
#define SENTINELCONTROL_DEVICEWAIT 0x3
#define SENTINELCONTROL_HOST 0x5
#define SENTINELCONTROL_HOSTRDY 0x6
#define SENTINELCONTROL_HOSTWAIT 0x7
// transfer
#define SENTINELCONTROL_TRAN 0x10
#define SENTINELCONTROL_TRANRDY 0x11
#define SENTINELCONTROL_TRANDONE 0x12
// transfer-method
#define SENTINELCONTROL_TRANSIZE 0x13
#define SENTINELCONTROL_TRANIN 0x14
#define SENTINELCONTROL_TRANOUT 0x15

#ifdef __cplusplus
};
#endif
#pragma endregion

#endif  /* _SENTINEL_H */