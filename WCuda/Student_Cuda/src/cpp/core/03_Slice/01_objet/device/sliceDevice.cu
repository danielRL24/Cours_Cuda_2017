#include "Indice1D.h"
#include "cudaTools.h"
#include "reductionADD.h"

#include <stdio.h>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/


/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/


/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void slice(float* ptrDevGM, int nbSlice);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static __device__ void reductionIntraThread(float* ptrDevTabSM, int nbSlice);
static __device__ float fpi(float x);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * output : void required !!
 */
__global__ void slice(float* ptrDevGM, int nbSlice)
    {
    extern __shared__ float ptrDevTabSM[];

    reductionIntraThread(ptrDevTabSM, nbSlice);

    __syncthreads();

    reductionADD<float>(ptrDevTabSM, ptrDevGM);
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__device__ void reductionIntraThread(float* ptrDevTabSM, int nbSlice)
    {
    const int NB_THREAD = blockDim.x * gridDim.x;
    const int TID = threadIdx.x + (blockIdx.x * blockDim.x);
    const int TID_LOCAL = threadIdx.x;
    const float DX = 1.f / (float) nbSlice;

    // Debug, facultatif (voir AddVector.cu)
//    if (TID == 0)
//	{
//	printf("Coucou from device tid = %d", TID); //required   Device::synchronize(); after the call of kernel
//	}

    float sumThread = 0;
    int s = TID;
    while (s < nbSlice)
	{
	sumThread += fpi(s * DX);
	s += NB_THREAD;
	}

    ptrDevTabSM[TID_LOCAL] = sumThread * DX;
    }

__device__ float fpi(float x)
    {
    return 4.f / (1.f + x * x);
    }
/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

