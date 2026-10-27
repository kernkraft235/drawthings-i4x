#include <stdint.h>
#include <stddef.h>
size_t ccv_nnc_8i_rowwise_x_data_size(int format, int datatype, size_t count, size_t width);
size_t ccv_nnc_quantize_8i_rowwise_x(const void*, int, int, size_t, size_t, int, const float*, size_t, const float*, size_t, void*, size_t);
void ccv_nnc_dequantize_8i_rowwise(const void*, int, int, size_t, size_t, void*, size_t);
void ccv_nnc_dequantize_8i_rowwise_x(const void*, int, int, size_t, size_t, int, void*, size_t);
int dt_decode_ezm7(const uint8_t* input, size_t length, uint16_t* output, size_t count);
int dt_finite(const void* input, int dtype, size_t count);
double dt_relative_rmse(const void* a, const void* b, int dtype, size_t rows, size_t width);
int dt_decode_palette(const uint8_t* input, size_t length, int dtype, int bits, void* output, size_t count);
