/* test_matmul_idot: the IDOT GEMM drivers (matmul_q_idot / matmul_i4_idot) vs a
 * plain-C reference, across S/O/I grids including every tiling remainder case.
 * On i8mm builds it additionally checks the SMMLA 2x2 tile path BITWISE against
 * the per-row SDOT path (integer math is exact: any difference is a bug). */
#define main coli_glm_main_unused
#include "../glm.c"
#undef main

#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static uint64_t rng = 0x243F6A8885A308D3ull;
static int8_t rnd8(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (int8_t)(rng & 0xFF); }

static void ref_q(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                  const float *scale, int S, int I, int O){
    for(int s=0;s<S;s++) for(int o=0;o<O;o++){
        int32_t acc=0; const int8_t *w=q+(int64_t)o*I; const int8_t *x=xq+(int64_t)s*I;
        for(int i=0;i<I;i++) acc+=(int32_t)w[i]*x[i];
        y[(int64_t)s*O+o]=(float)acc*scale[o]*sx[s];
    }
}
static void ref_i4(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                   const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    for(int s=0;s<S;s++) for(int o=0;o<O;o++){
        int32_t acc=0; const uint8_t *w=q4+(int64_t)o*rb; const int8_t *x=xq+(int64_t)s*I;
        for(int i=0;i<I;i++){ int v=(int)((i&1)?(w[i>>1]>>4):(w[i>>1]&0xF))-8; acc+=v*x[i]; }
        y[(int64_t)s*O+o]=(float)acc*scale[o]*sx[s];
    }
}

int main(void){
    const int Ss[]={1,2,3,4,5,7,16};
    const int Os[]={1,2,3,17,64};
    const int Is[]={1,15,16,17,31,32,33,64,257};
    enum { MAXS=16, MAXO=64, MAXI=257 };
    static int8_t  xq[MAXS*MAXI];
    static int8_t  q8[MAXO*MAXI];
    static uint8_t q4[MAXO*((MAXI+1)/2)];
    static float   sx[MAXS], sc[MAXO];
    static float   yr[MAXS*MAXO], yd[MAXS*MAXO], yt[MAXS*MAXO];

    for(int i=0;i<MAXS*MAXI;i++) xq[i]=rnd8();
    for(int i=0;i<MAXO*MAXI;i++) q8[i]=rnd8();
    q8[0]=-128; q8[1]=127;                      /* sign-trick / saturation edges */
    for(int i=0;i<MAXO*((MAXI+1)/2);i++) q4[i]=(uint8_t)rnd8();
    for(int i=0;i<MAXS;i++) sx[i]=0.01f*(float)(i+1);
    for(int i=0;i<MAXO;i++) sc[i]=0.001f*(float)(i+3);

    int cases=0;
    for(unsigned a=0;a<sizeof Ss/sizeof *Ss;a++)
    for(unsigned b=0;b<sizeof Os/sizeof *Os;b++)
    for(unsigned c=0;c<sizeof Is/sizeof *Is;c++){
        int S=Ss[a],O=Os[b],I=Is[c]; cases++;
        ref_q(yr,xq,sx,q8,sc,S,I,O);
        matmul_q_idot(yd,xq,sx,q8,sc,S,I,O);
        CHECK(memcmp(yr,yd,sizeof(float)*(size_t)S*O)==0);
        ref_i4(yr,xq,sx,q4,sc,S,I,O);
        matmul_i4_idot(yd,xq,sx,q4,sc,S,I,O);
        CHECK(memcmp(yr,yd,sizeof(float)*(size_t)S*O)==0);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
        /* bitwise: tile path vs per-row path */
        g_i8mm=1; matmul_q_idot(yt,xq,sx,q8,sc,S,I,O);
        g_i8mm=0; matmul_q_idot(yd,xq,sx,q8,sc,S,I,O);
        CHECK(memcmp(yt,yd,sizeof(float)*(size_t)S*O)==0);
        g_i8mm=1; matmul_i4_idot(yt,xq,sx,q4,sc,S,I,O);
        g_i8mm=0; matmul_i4_idot(yd,xq,sx,q4,sc,S,I,O);
        CHECK(memcmp(yt,yd,sizeof(float)*(size_t)S*O)==0);
        g_i8mm=1;
#endif
    }
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const char *mode="i8mm tile + per-row, bitwise";
#else
    const char *mode="per-row only (no i8mm on this build)";
#endif
    if(fails){ printf("matmul idot driver tests: %d FAILED (%d cases, %s)\n",fails,cases,mode); return 1; }
    printf("matmul idot driver tests: ok (%d cases, %s)\n",cases,mode);
    return 0;
}
