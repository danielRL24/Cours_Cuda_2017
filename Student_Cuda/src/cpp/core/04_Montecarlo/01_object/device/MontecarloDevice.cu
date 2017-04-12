#include "Indice1D.h"
#include "cudaTools.h"
#include "reductionADD.h"
#include "limits.h"

#include <curand_kernel.h>
#include <stdio.h>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void setup_kernel_rand(curandState* tabDevGenerator, int deviceID);
__global__ void montecarlo(int* ptrDevNx, curandState* tabDevGenerator, long n, float m);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static __device__ void reductionIntraThread(int* tabSM, curandState* tabDevGenerator, long n, float m);
__device__ float fpi(float x);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void setup_kernel_rand(curandState* tabDevGenerator, int deviceId)
    {
    // Customisation du generator:
    // Proposition, au lecteur de faire mieux !
    // Contrainte : Doit etre différent d'un GPU à l'autre
    // Contrainte : Doit etre différent d?un thread à l?autre
    const int TID = Indice1D::tid();
    int deltaSeed = deviceId * INT_MAX / 10000;
    int deltaSequence = deviceId * 100;
    int deltaOffset = deviceId * 100;
    int seed = 1234 + deltaSeed;
    int sequenceNumber = TID + deltaSequence;
    int offset = deltaOffset;
    curand_init(seed, sequenceNumber, offset, &tabDevGenerator[TID]);
    }

__global__ void montecarlo(int* ptrDevNx, curandState* tabDevGenerator, long n, float m)
    {
    extern __shared__ int tabSM[];

    reductionIntraThread(tabSM, tabDevGenerator, n, m);

    __syncthreads();

    reductionADD<int>(tabSM, ptrDevNx);
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__device__ void reductionIntraThread(int* tabSM, curandState* tabDevGenerator, long n, float m)
    {
    const int TID = Indice1D::tid();
    const int TID_LOCAL = Indice1D::tidLocal();
    // Global Memory -> Register (optimization)
    curandState localGenerator = tabDevGenerator[TID];
    float xAlea;
    float yAlea;
    int nx = 0;
    for (long i = 1; i <= n; i++)
	{
	xAlea = curand_uniform(&localGenerator);
	yAlea = curand_uniform(&localGenerator) * m;

	if(yAlea < fpi(xAlea))
	    {
	    nx += 1;
	    }
	}
    //Register -> Global Memory
    //Necessaire si on veut utiliser notre generator
    // - dans d?autre kernel
    // - avec d?autres nombres aleatoires !
    tabSM[TID_LOCAL] = nx;
    tabDevGenerator[TID] = localGenerator;
    }

__device__ float fpi(float x)
    {
    return 4.f / (1.f + x * x);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
