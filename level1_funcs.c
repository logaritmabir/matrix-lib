#include <stdio.h>
#include <stdlib.h>

int add(int n1, int n2){
    long res = n1 + n2;
    return res;
}
void saxpy(const float alpha, const int arr_length, const void* restrict xv, const void* restrict yv, void* restrict dstv){
    const float* x = (float*)xv;
    const float* y = (float*)yv;
    float* dst = (float*)dstv;
    for(int i = 0; i < arr_length; i++){
        dst[i] = x[i] * alpha + y[i];
    }
}