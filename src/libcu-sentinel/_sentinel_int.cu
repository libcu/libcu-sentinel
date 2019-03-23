#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <_sentinel_int.h>

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

#include "ext/mutex.cu"
#include "sentinel-gpu.cu"