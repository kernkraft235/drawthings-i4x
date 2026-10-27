#ifndef DT_CPU_COMPAT_H
#define DT_CPU_COMPAT_H
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#define CCV_WARN_UNUSED(x) x
#define CCV_32F 0x04000
#define CCV_64F 0x10000
#define CCV_16F 0x20000
#define CCV_16BF 0x80000
#define CCV_TENSOR_CPU_MEMORY 1
#define CCV_TENSOR_GPU_MEMORY 2
#define CCV_GET_DATA_TYPE_SIZE(x) ((x)==CCV_64F?8:(x)==CCV_32F?4:2)
#define ccmalloc malloc
#define cccalloc calloc
#define ccfree free
#define ccv_min(a,b) ((a)<(b)?(a):(b))
#define ccv_max(a,b) ((a)>(b)?(a):(b))
#define ccv_clamp(x,a,b) ccv_min(ccv_max((x),(a)),(b))
#define parallel_for(i,n) for (int i=0;i<(n);i++)
#define parallel_endfor
static inline void ccv_float_to_half_precision(const float* a, uint16_t* b, int n) { for(int i=0;i<n;i++){ _Float16 h=(_Float16)a[i]; memcpy(b+i,&h,2); } }
static inline void ccv_half_precision_to_float(const uint16_t* a, float* b, int n) { for(int i=0;i<n;i++){ _Float16 h; memcpy(&h,a+i,2); b[i]=(float)h; } }
static inline void ccv_float_to_bfloat(const float* a, uint16_t* b, int n) { for(int i=0;i<n;i++){ uint32_t u;memcpy(&u,a+i,4);u+=0x7fff+((u>>16)&1);b[i]=u>>16;} }
static inline void ccv_bfloat_to_float(const uint16_t* a, float* b, int n) { for(int i=0;i<n;i++){uint32_t u=(uint32_t)a[i]<<16;memcpy(b+i,&u,4);} }
/* Packed row-wise int8-compatible low-bit payload formats. */
enum {
	CCV_NNC_QX_8I_ROWWISE_Q4_K = 1,
	CCV_NNC_QX_8I_ROWWISE_Q3_K,
	CCV_NNC_QX_8I_ROWWISE_Q2_K,
	CCV_NNC_QX_8I_ROWWISE_IQ2_S,
	CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
	CCV_NNC_QX_8I_ROWWISE_IQ3_S,
	CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
	CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
	CCV_NNC_QX_8I_ROWWISE_Q5_K,
	CCV_NNC_QX_8I_ROWWISE_Q6_K,
	CCV_NNC_QX_8I_ROWWISE_IQ1_S,
	CCV_NNC_QX_8I_ROWWISE_IQ1_M,
};

/* Modifiers for the format argument / tensor reserved field of ROWWISE_X. */
enum {
	CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK = 0xff,
	/* W R_K, where R256 = H4^tensor4 / 16 and H4 is +1 except -1 on
	 * its anti-diagonal. Requires K % 256 == 0; no output rotation. */
	CCV_NNC_QX_8I_ROWWISE_HADAMARD_256 = 0x100,
};


static inline int ccv_nnc_8i_rowwise_x_group_size(const int format)
{
	switch (format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK)
	{
		case CCV_NNC_QX_8I_ROWWISE_Q5_K:
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
		case CCV_NNC_QX_8I_ROWWISE_IQ1_M:
			return 16;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
		case CCV_NNC_QX_8I_ROWWISE_IQ1_S:
			return 32;
		case CCV_NNC_QX_8I_ROWWISE_Q6_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			return 8;
		default:
			assert(0);
			return 0;
	}
}

static inline int ccv_nnc_8i_rowwise_x_group_bits(const int format)
{
	switch (format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK)
	{
		case CCV_NNC_QX_8I_ROWWISE_Q5_K:
			return 88;
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
			return 72;
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			return 56;
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
			return 42;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
			return 21;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
		case CCV_NNC_QX_8I_ROWWISE_IQ1_M:
			return 28;
		case CCV_NNC_QX_8I_ROWWISE_IQ1_S:
			return 50;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
			return 64;
		case CCV_NNC_QX_8I_ROWWISE_Q6_K:
			return 52;
		default:
			assert(0);
			return 0;
	}
}


#endif
