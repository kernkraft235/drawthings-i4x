#include "ccv_nnc.h"
void dt_upstream_depalettize(const void* input, const int datatype, const size_t input_length, const int qbits, const int number_in_blocks, void* output, const size_t output_length)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	const int num_blocks = (output_length + number_in_blocks - 1) / number_in_blocks;
	const size_t element_size = CCV_GET_DATA_TYPE_SIZE(datatype);
	uint8_t* const u8 = (uint8_t*)output;
	const uint8_t* const ui = (const uint8_t*)input;
	assert(qbits == 4 || qbits == 5 || qbits == 6 || qbits == 7 || qbits == 8);
	if (datatype == CCV_16F || datatype == CCV_16BF)
	{
		if (qbits == 4)
		{
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 16 + number_in_blocks / 2) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const uint16_t* const palette = (uint16_t*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 16;
				uint16_t* const f16 = (uint16_t*)u80;
				int j;
				if (nI % 2 == 0)
				{
					for (j = 0; j < nI; j += 2)
					{
						const uint8_t u0 = *ui1;
						const int i0 = (int)(u0 >> 4);
						const int i1 = (int)(u0 & 15);
						f16[j] = palette[i0];
						f16[j + 1] = palette[i1];
						++ui1;
					}
				} else {
					for (j = 0; j < nI; j += 2)
					{
						const uint8_t u0 = *ui1;
						const int i0 = (int)(u0 >> 4);
						const int i1 = (int)(u0 & 15);
						f16[j] = palette[i0];
						if (j + 1 < nI)
							f16[j + 1] = palette[i1];
						++ui1;
					}
				}
			} parallel_endfor
		} else if (qbits == 5) {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 32 + number_in_blocks / 8 * 5) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const uint16_t* const palette = (uint16_t*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 32;
				uint16_t* const f16 = (uint16_t*)u80;
				int j;
				if (nI % 8 == 0)
				{
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const int i0 = (int)(u0 >> 3);
						const int i1 = (int)(((u0 & 7) << 2) | (u1 >> 6));
						const int i2 = (int)((u1 >> 1) & 31);
						const int i3 = (int)(((u1 & 1) << 4) | (u2 >> 4));
						const int i4 = (int)(((u2 & 15) << 1) | (u3 >> 7));
						const int i5 = (int)((u3 >> 2) & 31);
						const int i6 = (int)(((u3 & 3) << 3) | (u4 >> 5));
						const int i7 = (int)(u4 & 31);
						f16[j] = palette[i0];
						f16[j + 1] = palette[i1];
						f16[j + 2] = palette[i2];
						f16[j + 3] = palette[i3];
						f16[j + 4] = palette[i4];
						f16[j + 5] = palette[i5];
						f16[j + 6] = palette[i6];
						f16[j + 7] = palette[i7];
						ui1 += 5;
					}
				} else {
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const int i0 = (int)(u0 >> 3);
						const int i1 = (int)(((u0 & 7) << 2) | (u1 >> 6));
						const int i2 = (int)((u1 >> 1) & 31);
						const int i3 = (int)(((u1 & 1) << 4) | (u2 >> 4));
						const int i4 = (int)(((u2 & 15) << 1) | (u3 >> 7));
						const int i5 = (int)((u3 >> 2) & 31);
						const int i6 = (int)(((u3 & 3) << 3) | (u4 >> 5));
						const int i7 = (int)(u4 & 31);
						f16[j] = palette[i0];
						if (j + 1 < nI)
							f16[j + 1] = palette[i1];
						if (j + 2 < nI)
							f16[j + 2] = palette[i2];
						if (j + 3 < nI)
							f16[j + 3] = palette[i3];
						if (j + 4 < nI)
							f16[j + 4] = palette[i4];
						if (j + 5 < nI)
							f16[j + 5] = palette[i5];
						if (j + 6 < nI)
							f16[j + 6] = palette[i6];
						if (j + 7 < nI)
							f16[j + 7] = palette[i7];
						ui1 += 5;
					}
				}
			} parallel_endfor
		} else if (qbits == 6) {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 64 + number_in_blocks / 4 * 3) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const uint16_t* const palette = (uint16_t*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 64;
				uint16_t* const f16 = (uint16_t*)u80;
				int j;
				if (nI % 4 == 0)
				{
					for (j = 0; j < nI; j += 4)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const int i0 = (int)(u0 >> 2);
						const int i1 = (int)(((u0 & 3) << 4) | (u1 >> 4));
						const int i2 = (int)(((u1 & 15) << 2) | (u2 >> 6));
						const int i3 = (int)(u2 & 63);
						f16[j] = palette[i0];
						f16[j + 1] = palette[i1];
						f16[j + 2] = palette[i2];
						f16[j + 3] = palette[i3];
						ui1 += 3;
					}
				} else {
					for (j = 0; j < nI; j += 4)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const int i0 = (int)(u0 >> 2);
						const int i1 = (int)(((u0 & 3) << 4) | (u1 >> 4));
						const int i2 = (int)(((u1 & 15) << 2) | (u2 >> 6));
						const int i3 = (int)(u2 & 63);
						f16[j] = palette[i0];
						if (j + 1 < nI)
							f16[j + 1] = palette[i1];
						if (j + 2 < nI)
							f16[j + 2] = palette[i2];
						if (j + 3 < nI)
							f16[j + 3] = palette[i3];
						ui1 += 3;
					}
				}
			} parallel_endfor
		} else if (qbits == 7) {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 128 + number_in_blocks / 8 * 7) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const uint16_t* const palette = (uint16_t*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 128;
				uint16_t* const f16 = (uint16_t*)u80;
				int j;
				if (nI % 8 == 0)
				{
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const uint8_t u5 = ui1[5];
						const uint8_t u6 = ui1[6];
						const int i0 = (int)(u0 >> 1);
						const int i1 = (int)(((u0 & 1) << 6) | (u1 >> 2));
						const int i2 = (int)(((u1 & 3) << 5) | (u2 >> 3));
						const int i3 = (int)(((u2 & 7) << 4) | (u3 >> 4));
						const int i4 = (int)(((u3 & 15) << 3) | (u4 >> 5));
						const int i5 = (int)(((u4 & 31) << 2) | (u5 >> 6));
						const int i6 = (int)(((u5 & 63) << 1) | (u6 >> 7));
						const int i7 = (int)(u6 & 127);
						f16[j] = palette[i0];
						f16[j + 1] = palette[i1];
						f16[j + 2] = palette[i2];
						f16[j + 3] = palette[i3];
						f16[j + 4] = palette[i4];
						f16[j + 5] = palette[i5];
						f16[j + 6] = palette[i6];
						f16[j + 7] = palette[i7];
						ui1 += 7;
					}
				} else {
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const uint8_t u5 = ui1[5];
						const uint8_t u6 = ui1[6];
						const int i0 = (int)(u0 >> 1);
						const int i1 = (int)(((u0 & 1) << 6) | (u1 >> 2));
						const int i2 = (int)(((u1 & 3) << 5) | (u2 >> 3));
						const int i3 = (int)(((u2 & 7) << 4) | (u3 >> 4));
						const int i4 = (int)(((u3 & 15) << 3) | (u4 >> 5));
						const int i5 = (int)(((u4 & 31) << 2) | (u5 >> 6));
						const int i6 = (int)(((u5 & 63) << 1) | (u6 >> 7));
						const int i7 = (int)(u6 & 127);
						f16[j] = palette[i0];
						if (j + 1 < nI)
							f16[j + 1] = palette[i1];
						if (j + 2 < nI)
							f16[j + 2] = palette[i2];
						if (j + 3 < nI)
							f16[j + 3] = palette[i3];
						if (j + 4 < nI)
							f16[j + 4] = palette[i4];
						if (j + 5 < nI)
							f16[j + 5] = palette[i5];
						if (j + 6 < nI)
							f16[j + 6] = palette[i6];
						if (j + 7 < nI)
							f16[j + 7] = palette[i7];
						ui1 += 7;
					}
				}
			} parallel_endfor
		} else {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 256 + number_in_blocks) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const uint16_t* const palette = (uint16_t*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 256;
				uint16_t* const f16 = (uint16_t*)u80;
				int j;
				for (j = 0; j < nI; j++)
				{
					const uint8_t u0 = *ui1;
					f16[j] = palette[u0];
					++ui1;
				}
			} parallel_endfor
		}
	} else if (datatype == CCV_32F) {
		if (qbits == 4)
		{
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 16 + number_in_blocks / 2) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const float* const palette = (float*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 16;
				float* const f32 = (float*)u80;
				int j;
				if (nI % 2 == 0)
				{
					for (j = 0; j < nI; j += 2)
					{
						const uint8_t u0 = *ui1;
						const int i0 = (int)(u0 >> 4);
						const int i1 = (int)(u0 & 15);
						f32[j] = palette[i0];
						f32[j + 1] = palette[i1];
						++ui1;
					}
				} else {
					for (j = 0; j < nI; j += 2)
					{
						const uint8_t u0 = *ui1;
						const int i0 = (int)(u0 >> 4);
						const int i1 = (int)(u0 & 15);
						f32[j] = palette[i0];
						if (j + 1 < nI)
							f32[j + 1] = palette[i1];
						++ui1;
					}
				}
			} parallel_endfor
		} else if (qbits == 5) {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 32 + number_in_blocks / 8 * 5) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const float* const palette = (float*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 32;
				float* const f32 = (float*)u80;
				int j;
				if (nI % 8 == 0)
				{
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const int i0 = (int)(u0 >> 3);
						const int i1 = (int)(((u0 & 7) << 2) | (u1 >> 6));
						const int i2 = (int)((u1 >> 1) & 31);
						const int i3 = (int)(((u1 & 1) << 4) | (u2 >> 4));
						const int i4 = (int)(((u2 & 15) << 1) | (u3 >> 7));
						const int i5 = (int)((u3 >> 2) & 31);
						const int i6 = (int)(((u3 & 3) << 3) | (u4 >> 5));
						const int i7 = (int)(u4 & 31);
						f32[j] = palette[i0];
						f32[j + 1] = palette[i1];
						f32[j + 2] = palette[i2];
						f32[j + 3] = palette[i3];
						f32[j + 4] = palette[i4];
						f32[j + 5] = palette[i5];
						f32[j + 6] = palette[i6];
						f32[j + 7] = palette[i7];
						ui1 += 5;
					}
				} else {
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const int i0 = (int)(u0 >> 3);
						const int i1 = (int)(((u0 & 7) << 2) | (u1 >> 6));
						const int i2 = (int)((u1 >> 1) & 31);
						const int i3 = (int)(((u1 & 1) << 4) | (u2 >> 4));
						const int i4 = (int)(((u2 & 15) << 1) | (u3 >> 7));
						const int i5 = (int)((u3 >> 2) & 31);
						const int i6 = (int)(((u3 & 3) << 3) | (u4 >> 5));
						const int i7 = (int)(u4 & 31);
						f32[j] = palette[i0];
						if (j + 1 < nI)
							f32[j + 1] = palette[i1];
						if (j + 2 < nI)
							f32[j + 2] = palette[i2];
						if (j + 3 < nI)
							f32[j + 3] = palette[i3];
						if (j + 4 < nI)
							f32[j + 4] = palette[i4];
						if (j + 5 < nI)
							f32[j + 5] = palette[i5];
						if (j + 6 < nI)
							f32[j + 6] = palette[i6];
						if (j + 7 < nI)
							f32[j + 7] = palette[i7];
						ui1 += 5;
					}
				}
			} parallel_endfor
		} else if (qbits == 6) {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 64 + number_in_blocks / 4 * 3) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const float* const palette = (float*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 64;
				float* const f32 = (float*)u80;
				int j;
				if (nI % 4 == 0)
				{
					for (j = 0; j < nI; j += 4)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const int i0 = (int)(u0 >> 2);
						const int i1 = (int)(((u0 & 3) << 4) | (u1 >> 4));
						const int i2 = (int)(((u1 & 15) << 2) | (u2 >> 6));
						const int i3 = (int)(u2 & 63);
						f32[j] = palette[i0];
						f32[j + 1] = palette[i1];
						f32[j + 2] = palette[i2];
						f32[j + 3] = palette[i3];
						ui1 += 3;
					}
				} else {
					for (j = 0; j < nI; j += 4)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const int i0 = (int)(u0 >> 2);
						const int i1 = (int)(((u0 & 3) << 4) | (u1 >> 4));
						const int i2 = (int)(((u1 & 15) << 2) | (u2 >> 6));
						const int i3 = (int)(u2 & 63);
						f32[j] = palette[i0];
						if (j + 1 < nI)
							f32[j + 1] = palette[i1];
						if (j + 2 < nI)
							f32[j + 2] = palette[i2];
						if (j + 3 < nI)
							f32[j + 3] = palette[i3];
						ui1 += 3;
					}
				}
			} parallel_endfor
		} else if (qbits == 7) {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 128 + number_in_blocks / 8 * 7) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const float* const palette = (float*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 128;
				float* const f32 = (float*)u80;
				int j;
				if (nI % 8 == 0)
				{
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const uint8_t u5 = ui1[5];
						const uint8_t u6 = ui1[6];
						const int i0 = (int)(u0 >> 1);
						const int i1 = (int)(((u0 & 1) << 6) | (u1 >> 2));
						const int i2 = (int)(((u1 & 3) << 5) | (u2 >> 3));
						const int i3 = (int)(((u2 & 7) << 4) | (u3 >> 4));
						const int i4 = (int)(((u3 & 15) << 3) | (u4 >> 5));
						const int i5 = (int)(((u4 & 31) << 2) | (u5 >> 6));
						const int i6 = (int)(((u5 & 63) << 1) | (u6 >> 7));
						const int i7 = (int)(u6 & 127);
						f32[j] = palette[i0];
						f32[j + 1] = palette[i1];
						f32[j + 2] = palette[i2];
						f32[j + 3] = palette[i3];
						f32[j + 4] = palette[i4];
						f32[j + 5] = palette[i5];
						f32[j + 6] = palette[i6];
						f32[j + 7] = palette[i7];
						ui1 += 7;
					}
				} else {
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const uint8_t u5 = ui1[5];
						const uint8_t u6 = ui1[6];
						const int i0 = (int)(u0 >> 1);
						const int i1 = (int)(((u0 & 1) << 6) | (u1 >> 2));
						const int i2 = (int)(((u1 & 3) << 5) | (u2 >> 3));
						const int i3 = (int)(((u2 & 7) << 4) | (u3 >> 4));
						const int i4 = (int)(((u3 & 15) << 3) | (u4 >> 5));
						const int i5 = (int)(((u4 & 31) << 2) | (u5 >> 6));
						const int i6 = (int)(((u5 & 63) << 1) | (u6 >> 7));
						const int i7 = (int)(u6 & 127);
						f32[j] = palette[i0];
						if (j + 1 < nI)
							f32[j + 1] = palette[i1];
						if (j + 2 < nI)
							f32[j + 2] = palette[i2];
						if (j + 3 < nI)
							f32[j + 3] = palette[i3];
						if (j + 4 < nI)
							f32[j + 4] = palette[i4];
						if (j + 5 < nI)
							f32[j + 5] = palette[i5];
						if (j + 6 < nI)
							f32[j + 6] = palette[i6];
						if (j + 7 < nI)
							f32[j + 7] = palette[i7];
						ui1 += 7;
					}
				}
			} parallel_endfor
		} else {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 256 + number_in_blocks) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const float* const palette = (float*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 256;
				float* const f32 = (float*)u80;
				int j;
				for (j = 0; j < nI; j++)
				{
					const uint8_t u0 = *ui1;
					f32[j] = palette[u0];
					++ui1;
				}
			} parallel_endfor
		}
	} else {
		if (qbits == 4)
		{
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 16 + number_in_blocks / 2) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const double* const palette = (double*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 16;
				double* const f64 = (double*)u80;
				int j;
				if (nI % 2 == 0)
				{
					for (j = 0; j < nI; j += 2)
					{
						const uint8_t u0 = *ui1;
						const int i0 = (int)(u0 >> 4);
						const int i1 = (int)(u0 & 15);
						f64[j] = palette[i0];
						f64[j + 1] = palette[i1];
						++ui1;
					}
				} else {
					for (j = 0; j < nI; j += 2)
					{
						const uint8_t u0 = *ui1;
						const int i0 = (int)(u0 >> 4);
						const int i1 = (int)(u0 & 15);
						f64[j] = palette[i0];
						if (j + 1 < nI)
							f64[j + 1] = palette[i1];
						++ui1;
					}
				}
			} parallel_endfor
		} else if (qbits == 5) {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 32 + number_in_blocks / 8 * 5) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const double* const palette = (double*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 32;
				double* const f64 = (double*)u80;
				int j;
				if (nI % 8 == 0)
				{
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const int i0 = (int)(u0 >> 3);
						const int i1 = (int)(((u0 & 7) << 2) | (u1 >> 6));
						const int i2 = (int)((u1 >> 1) & 31);
						const int i3 = (int)(((u1 & 1) << 4) | (u2 >> 4));
						const int i4 = (int)(((u2 & 15) << 1) | (u3 >> 7));
						const int i5 = (int)((u3 >> 2) & 31);
						const int i6 = (int)(((u3 & 3) << 3) | (u4 >> 5));
						const int i7 = (int)(u4 & 31);
						f64[j] = palette[i0];
						f64[j + 1] = palette[i1];
						f64[j + 2] = palette[i2];
						f64[j + 3] = palette[i3];
						f64[j + 4] = palette[i4];
						f64[j + 5] = palette[i5];
						f64[j + 6] = palette[i6];
						f64[j + 7] = palette[i7];
						ui1 += 5;
					}
				} else {
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const int i0 = (int)(u0 >> 3);
						const int i1 = (int)(((u0 & 7) << 2) | (u1 >> 6));
						const int i2 = (int)((u1 >> 1) & 31);
						const int i3 = (int)(((u1 & 1) << 4) | (u2 >> 4));
						const int i4 = (int)(((u2 & 15) << 1) | (u3 >> 7));
						const int i5 = (int)((u3 >> 2) & 31);
						const int i6 = (int)(((u3 & 3) << 3) | (u4 >> 5));
						const int i7 = (int)(u4 & 31);
						f64[j] = palette[i0];
						if (j + 1 < nI)
							f64[j + 1] = palette[i1];
						if (j + 2 < nI)
							f64[j + 2] = palette[i2];
						if (j + 3 < nI)
							f64[j + 3] = palette[i3];
						if (j + 4 < nI)
							f64[j + 4] = palette[i4];
						if (j + 5 < nI)
							f64[j + 5] = palette[i5];
						if (j + 6 < nI)
							f64[j + 6] = palette[i6];
						if (j + 7 < nI)
							f64[j + 7] = palette[i7];
						ui1 += 5;
					}
				}
			} parallel_endfor
		} else if (qbits == 6) {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 64 + number_in_blocks / 4 * 3) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const double* const palette = (double*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 64;
				double* const f64 = (double*)u80;
				int j;
				if (nI % 4 == 0)
				{
					for (j = 0; j < nI; j += 4)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const int i0 = (int)(u0 >> 2);
						const int i1 = (int)(((u0 & 3) << 4) | (u1 >> 4));
						const int i2 = (int)(((u1 & 15) << 2) | (u2 >> 6));
						const int i3 = (int)(u2 & 63);
						f64[j] = palette[i0];
						f64[j + 1] = palette[i1];
						f64[j + 2] = palette[i2];
						f64[j + 3] = palette[i3];
						ui1 += 3;
					}
				} else {
					for (j = 0; j < nI; j += 4)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const int i0 = (int)(u0 >> 2);
						const int i1 = (int)(((u0 & 3) << 4) | (u1 >> 4));
						const int i2 = (int)(((u1 & 15) << 2) | (u2 >> 6));
						const int i3 = (int)(u2 & 63);
						f64[j] = palette[i0];
						if (j + 1 < nI)
							f64[j + 1] = palette[i1];
						if (j + 2 < nI)
							f64[j + 2] = palette[i2];
						if (j + 3 < nI)
							f64[j + 3] = palette[i3];
						ui1 += 3;
					}
				}
			} parallel_endfor
		} else if (qbits == 7) {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 128 + number_in_blocks / 8 * 7) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const double* const palette = (double*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 128;
				double* const f64 = (double*)u80;
				int j;
				if (nI % 8 == 0)
				{
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const uint8_t u5 = ui1[5];
						const uint8_t u6 = ui1[6];
						const int i0 = (int)(u0 >> 1);
						const int i1 = (int)(((u0 & 1) << 6) | (u1 >> 2));
						const int i2 = (int)(((u1 & 3) << 5) | (u2 >> 3));
						const int i3 = (int)(((u2 & 7) << 4) | (u3 >> 4));
						const int i4 = (int)(((u3 & 15) << 3) | (u4 >> 5));
						const int i5 = (int)(((u4 & 31) << 2) | (u5 >> 6));
						const int i6 = (int)(((u5 & 63) << 1) | (u6 >> 7));
						const int i7 = (int)(u6 & 127);
						f64[j] = palette[i0];
						f64[j + 1] = palette[i1];
						f64[j + 2] = palette[i2];
						f64[j + 3] = palette[i3];
						f64[j + 4] = palette[i4];
						f64[j + 5] = palette[i5];
						f64[j + 6] = palette[i6];
						f64[j + 7] = palette[i7];
						ui1 += 7;
					}
				} else {
					for (j = 0; j < nI; j += 8)
					{
						const uint8_t u0 = ui1[0];
						const uint8_t u1 = ui1[1];
						const uint8_t u2 = ui1[2];
						const uint8_t u3 = ui1[3];
						const uint8_t u4 = ui1[4];
						const uint8_t u5 = ui1[5];
						const uint8_t u6 = ui1[6];
						const int i0 = (int)(u0 >> 1);
						const int i1 = (int)(((u0 & 1) << 6) | (u1 >> 2));
						const int i2 = (int)(((u1 & 3) << 5) | (u2 >> 3));
						const int i3 = (int)(((u2 & 7) << 4) | (u3 >> 4));
						const int i4 = (int)(((u3 & 15) << 3) | (u4 >> 5));
						const int i5 = (int)(((u4 & 31) << 2) | (u5 >> 6));
						const int i6 = (int)(((u5 & 63) << 1) | (u6 >> 7));
						const int i7 = (int)(u6 & 127);
						f64[j] = palette[i0];
						if (j + 1 < nI)
							f64[j + 1] = palette[i1];
						if (j + 2 < nI)
							f64[j + 2] = palette[i2];
						if (j + 3 < nI)
							f64[j + 3] = palette[i3];
						if (j + 4 < nI)
							f64[j + 4] = palette[i4];
						if (j + 5 < nI)
							f64[j + 5] = palette[i5];
						if (j + 6 < nI)
							f64[j + 6] = palette[i6];
						if (j + 7 < nI)
							f64[j + 7] = palette[i7];
						ui1 += 7;
					}
				}
			} parallel_endfor
		} else {
			parallel_for(i, num_blocks) {
				const int nI = ccv_min(number_in_blocks, output_length - i * number_in_blocks);
				const uint8_t* const ui0 = ui + (element_size * 256 + number_in_blocks) * i;
				uint8_t* const u80 = u8 + element_size * number_in_blocks * i;
				const double* const palette = (double*)ui0;
				const uint8_t* ui1 = ui0 + element_size * 256;
				double* const f64 = (double*)u80;
				int j;
				for (j = 0; j < nI; j++)
				{
					const uint8_t u0 = *ui1;
					f64[j] = palette[u0];
					++ui1;
				}
			} parallel_endfor
		}
	}
}
