#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sentinel.h>

#ifdef __cplusplus
extern "C" {;
#endif

//////////////////////
// RUNTIME
#pragma region RUNTIME

bool gpuAssert(cudaError_t code, const char *action, const char *file, int line, bool abort) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s [%s:%d]\n", cudaGetErrorString(code), file, line);
		//getchar();
		if (abort) exit(code);
		return false;
	}
	return true;
}

#if __CUDA_ARCH__
__device__ void usleep(int time) {
}
#endif

/* Compare S1 and S2.  */
__device__ int strcmp_(const char *s1, const char *s2) {
#ifndef OMIT_PTX
	int r;
	asm(
		".reg .pred p1;\n\t"
		".reg .u32 c1, c2;\n\t"

		//
		"_While0:\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"ld.u8 			c2, [%2];\n\t"
		"@!p1 bra _Ret;\n\t"
		"setp.eq.u32	p1, c1, c2;\n\t"
		"@!p1 bra _Ret;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		"sub.u32 		%0, c1, c2;\n\t"
		: "=" __I(r) : __R(s1), __R(s2));
	return r;
#else
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (*a && *a == *b) { a++; b++; }
	return *a - *b;
#endif
}

/* Return the length of S.  */
__device__ size_t strlen_(const char *s) {
#ifndef OMIT_PTX
	size_t r;
	asm(
		".reg .pred p1;\n\t"
		".reg " _UX " s2;\n\t"
		".reg " _BX " r;\n\t"
		".reg .b16 c;\n\t"
		"mov" _BX "		%0, 0;\n\t"

		"setp.eq" _UX "	p1, %1, 0;\n\t"
		"@p1 bra _End;\n\t"
		"mov" _UX "		s2, %1;\n\t"

		"_While:\n\t"
		"ld.u8			c, [s2];\n\t"
		//"and.b16		c, c, 255;\n\t"
		"setp.ne.u16	p1, c, 0;\n\t"
		"@!p1 bra _Value;\n\t"
		"add" _UX "		s2, s2, 1;\n\t"
		"bra.uni _While;\n\t"

		"_Value:\n\t"
		"sub" _UX "		r, s2, %1;\n\t"
		"and" _BX "		%0, r, 0x3fffffff;\n\t"
		"_End:\n\t"
		: "=" __R(r) : __R(s));
	return r;
#else
	if (!s) return 0;
	register const char *s2 = s;
	while (*s2) { s2++; }
	return 0x3fffffff & (int)(s2 - s);
#endif
}

#pragma endregion

//////////////////////
// MUTEX
#pragma region MUTEX

#if __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#define SLEEP(MS) __nanosleep(MS * 1000)
#else
#define SLEEP(MS) usleep(MS * 1000);
#endif
#elif __OS_WIN
#define SLEEP(MS) Sleep(MS)
#elif __OS_UNIX
#define SLEEP(MS) sleep(MS)
#endif

/* Mutex default sleep */
__hostb_device__ mutexSleep_t mutexDefaultSleep = { 0, 256, .5 };

/* Mutex with exponential back-off. */
__host_device__ void mutexSpinLock(void **cancelToken, volatile long *mutex, long cmp, long val, char pred, long predVal, bool(*func)(void **), void **funcTag, mutexSleep_t *ms) {
	if (ms == nullptr) ms = &mutexDefaultSleep;
	long v; ms->ms = ms->msmin;
#if __CUDA_ARCH__
	while ((!cancelToken || *cancelToken) && (v = atomicCAS((int *)mutex, cmp, val)) != cmp) {
#elif __OS_WIN
	while ((!cancelToken || *cancelToken) && (v = _InterlockedCompareExchange((volatile long *)mutex, cmp, val)) != cmp) {
#elif __OS_UNIX
	while ((!cancelToken || *cancelToken) && (v = __sync_val_compare_and_swap((long *)mutex, cmp, val)) != cmp) {
#endif
		bool condition = false;
		switch (pred) {
		case MUTEXPRED_EQ: condition = v == predVal; break;
		case MUTEXPRED_NE: condition = v != predVal; break;
		case MUTEXPRED_LT: condition = v < predVal; break;
		case MUTEXPRED_GT: condition = v > predVal; break;
		case MUTEXPRED_LTE: condition = v <= predVal; break;
		case MUTEXPRED_GTE: condition = v >= predVal; break;
		case MUTEXPRED_AND: condition = v & predVal; break;
		case MUTEXPRED_ANE: condition = (v & predVal) == predVal; break;
		}
		if (condition && (!func || !func(funcTag))) return;
		SLEEP((int)ms->ms);
		ms->ms = ms->ms <= 0 ? ms->factor :
			ms->ms < ms->msmax ? ms->ms * ms->factor :
			ms->msmax;
	}
}

/* Mutex set. */
__host_device__ void mutexSet(volatile long *mutex, long val) {
#if __CUDA_ARCH__
	atomicExch((int *)mutex, val);
#elif __OS_WIN
	_InterlockedExchange((volatile long *)mutex, val);
#elif __OS_UNIX
	__sync_lock_test_and_set((long *)mutex, val);
#endif
}

#pragma endregion

//////////////////////
// SENTINEL
#pragma region SENTINEL

#if HAS_DEVICESENTINEL

static __device__ void executeTrans(char id, sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans);

static __device__ char *preparePtrs(sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut, sentinelCommand *cmd, char *data, char *dataEnd, intptr_t offset, sentinelOutPtr *&listOut_, char *&trans) {
	sentinelInPtr *i; sentinelOutPtr *o; char **field; char *ptr = data, *next;
	// PREPARE & TRANSFER
	int transSize = 0;
	sentinelInPtr *listIn = nullptr;
	if (ptrsIn)
		for (i = ptrsIn, field = (char **)i->field; field; i++, field = (char **)i->field) {
			if (!*field) continue;
			int size = i->size != -1 ? i->size : (i->size = (int)strlen(*field) + 1);
			next = ptr + size;
			if (!size) *field = nullptr;
			else if (next <= dataEnd) { i->unknown = ptr; ptr = next; }
			else { i->unknown = listIn; listIn = i; transSize += size; }
		}
	sentinelOutPtr *listOut = nullptr;
	if (ptrsOut) {
		if (ptrsOut[0].field != (char *)-1) ptr = data;
		else ptrsOut++; // { -1 } = append
		for (o = ptrsOut, field = (char **)o->field; field; o++, field = (char **)o->field) {
			int size = o->size != -1 ? o->size : (o->size = (int)(dataEnd - ptr));
			next = ptr + size;
			if (!size) *field = nullptr;
			else if (next <= dataEnd) { *field = ptr + offset; o->unknown = (void *)-1; ptr = next; }
			else { o->unknown = listOut; listOut = o; transSize += size; continue; }
		}
	}
	listOut_ = listOut;

	// TRANSFER & PACK
	if (transSize)
		executeTrans(0, cmd, transSize, listIn, listOut, offset, trans); // size & transfer-in
	if (ptrsIn)
		for (i = ptrsIn, field = (char **)i->field; field; i++, field = (char **)i->field) {
			if (!*field || !(ptr = (char *)i->unknown)) continue;
			memcpy(ptr, *field, i->size);
			*field = ptr + offset;
		}
	return data;
}

static __device__ bool postfixPtrs(sentinelOutPtr *ptrsOut, sentinelCommand *cmd, intptr_t offset, sentinelOutPtr *listOut, char *&trans) {
	sentinelOutPtr *o; char **field, **buf;
	// UNPACK & TRANSFER
	if (ptrsOut) {
		if (ptrsOut[0].field == (char *)-1) ptrsOut++; // { -1 } = append
		for (o = ptrsOut, field = (char **)o->field; field; o++, field = (char **)o->field) {
			if (o->unknown != (void *)-1) continue;
			if (!*field || !(buf = (char **)o->buf)) continue;
			int size = !o->sizeField ? o->size : *(int *)o->sizeField;
			if (size > 0) memcpy(*buf, *field - offset, size);
		}
	}
	if (listOut)
		executeTrans(1, cmd, 0, nullptr, listOut, offset, trans);
	return true;
}

__device__ volatile unsigned int _sentinelMapId;
__constant__ const sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
__device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut) {
	const sentinelMap *map = _sentinelDeviceMap[_sentinelMapId++ % SENTINEL_DEVICEMAPS];
	if (!map)
		panic("sentinel: device map not defined. did you start sentinel?\n");

	// ATTACH
	long id = atomicAdd((int *)&map->setId, 1);
	sentinelCommand *cmd = (sentinelCommand *)&map->cmds[id % SENTINEL_MSGCOUNT];
	if (cmd->magic != SENTINEL_MAGIC)
		panic("bad sentinel magic");
	atomicAdd(&cmd->locks, 1);
	volatile long *control = &cmd->control; intptr_t offset = map->offset; char *trans = nullptr;
	mutexSpinLock(nullptr, control, SENTINELCONTROL_NORMAL, SENTINELCONTROL_DEVICE);
	if (cmd->locks != 1)
		panic("bad sentinel lock");

	// PREPARE
	char *data = cmd->data + ROUND8_(msgLength), *dataEnd = data + msg->size;
	sentinelOutPtr *listOut = nullptr;
	if (((ptrsIn || ptrsOut) && !(data = preparePtrs(ptrsIn, ptrsOut, cmd, data, dataEnd, offset, listOut, trans))) ||
		(msg->prepare && !msg->prepare(msg, data, dataEnd, offset)))
		panic("msg too long");
	if (listOut)
		msg->flow |= SENTINELFLOW_TRAN;
	cmd->length = msgLength; memcpy(cmd->data, msg, msgLength);
	//printf("msg: %d[%d]'", msg->op, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	mutexSet(control, SENTINELCONTROL_DEVICERDY);

	// FLOW-WAIT
	if (msg->flow & SENTINELFLOW_WAIT) {
		mutexSpinLock(nullptr, control, SENTINELCONTROL_HOSTRDY, SENTINELCONTROL_DEVICEWAIT);
		cmd->length = msgLength; memcpy(msg, cmd->data, msgLength);
		if ((ptrsOut && !postfixPtrs(ptrsOut, cmd, offset, listOut, trans)) ||
			(msg->postfix && !msg->postfix(msg, offset)))
			panic("postfix error");
		mutexSet(control, !listOut ? SENTINELCONTROL_NORMAL : SENTINELCONTROL_DEVICERDY);
	}
	atomicSub(&cmd->locks, 1);
}

static __device__ void executeTrans(char id, sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans) {
	volatile long *control = &cmd->control;
	sentinelInPtr *i; sentinelOutPtr *o; char **field; char *data = cmd->data, *ptr = trans;
	switch (id) {
	case 0:
		cmd->length = size;
		mutexSet(control, SENTINELCONTROL_TRANSIZE);
		mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRAN);
		ptr = trans = *(char **)data;
		if (listIn)
			for (i = listIn; i; i = (sentinelInPtr *)i->unknown) {
				field = (char **)i->field;
				const char *v = (const char *)*field; int remain = i->size, length = 0;
				while (remain > 0) {
					length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
					memcpy(data, (void *)v, length); remain -= length; v += length;
					mutexSet(control, SENTINELCONTROL_TRANIN);
					mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRAN);
				}
				*field = ptr; ptr += i->size;
				i->unknown = nullptr;
			}
		if (listOut) {
			for (o = listOut; o; o = (sentinelOutPtr *)o->unknown) {
				field = (char **)o->field;
				*field = ptr; ptr += o->size;
			}
		}
		break;
	case 1:
		if (listOut)
			for (o = listOut; o; o = (sentinelOutPtr *)o->unknown) {
				field = (char **)o->buf;
				const char *v = (const char *)*field; int remain = o->size, length = 0;
				while (remain > 0) {
					length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
					mutexSet(control, SENTINELCONTROL_TRANOUT);
					mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRAN);
					memcpy((void *)v, data, length); remain -= length; v += length;
				}
				o->unknown = nullptr;
			}
		break;
	}
}

#endif

#pragma endregion

#ifdef __cplusplus
};
#endif