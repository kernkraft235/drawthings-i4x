#include "bridge.h"
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
void dt_upstream_depalettize(const void*,int,size_t,int,int,void*,size_t);

int dt_decode_palette(const uint8_t* input,size_t length,int dtype,int bits,void* output,size_t count) {
  if(length<4 || bits<4 || bits>8) return 0;
  int32_t block;memcpy(&block,input,4);
  int group=(bits==5 || bits==7)?8:bits==6?4:1;
  if(block<=0 || block%8 || count%group) return 0;
  size_t element=dtype==0x10000?8:dtype==0x4000?4:2;
  size_t blocks=(count+block-1)/block;
  size_t expected=blocks*((size_t)1<<bits)*element+(count*bits+7)/8;
  if(length!=4+expected || blocks>INT_MAX) return 0;
  dt_upstream_depalettize(input+4,dtype,expected,bits,block,output,count);
  return 1;
}

static double value(const void* p, int dtype, size_t i) {
  if (dtype == 0x20000) { _Float16 h; memcpy(&h, (const uint8_t*)p+i*2,2); return (double)h; }
  if (dtype == 0x4000) return ((const float*)p)[i];
  if (dtype == 0x10000) return ((const double*)p)[i];
  uint32_t bits=(uint32_t)((const uint16_t*)p)[i]<<16; float f; memcpy(&f,&bits,4); return f;
}
int dt_finite(const void* input,int dtype,size_t count) {
  for(size_t i=0;i<count;i++) if(!isfinite(value(input,dtype,i))) return 0;
  return 1;
}
double dt_relative_rmse(const void* a,const void* b,int dtype,size_t rows,size_t width) {
  size_t samples=rows<64?rows:64; double error=0,norm=0;
  for(size_t s=0;s<samples;s++) {
    size_t row=samples==1?0:s*(rows-1)/(samples-1);
    for(size_t j=0;j<width;j++) {size_t i=row*width+j;double x=value(a,dtype,i),y=value(b,dtype,i);error+=(y-x)*(y-x);norm+=x*x;}
  }
  return sqrt(error/fmax(norm,1e-60));
}
int dt_decode_ezm7(const uint8_t* input,size_t length,uint16_t* output,size_t count) {
  if(length<4 || count>UINT_MAX) return 0;
  int32_t n;memcpy(&n,input,4);
  if(n<0 || (size_t)n+4+count!=length) return 0;
  uint8_t* exponents=malloc(count); if(!exponents) return 0;
  int ok=0;
  for(int wrap=0;wrap<2 && !ok;wrap++) {
    z_stream z={0}; z.next_in=(Bytef*)input+4;z.avail_in=n;z.next_out=exponents;z.avail_out=(uInt)count;
    if(inflateInit2(&z,wrap?15:-15)!=Z_OK) continue;
    int r=inflate(&z,Z_FINISH);ok=r==Z_STREAM_END && z.total_out==count;inflateEnd(&z);
  }
  if(ok) for(size_t i=0;i<count;i++) {
    if(exponents[i]>31) {ok=0;break;}
    uint16_t m=input[4+n+i];output[i]=((m>>7)<<15)|((uint16_t)exponents[i]<<10)|((m&127)<<3);
  }
  free(exponents);return ok;
}
