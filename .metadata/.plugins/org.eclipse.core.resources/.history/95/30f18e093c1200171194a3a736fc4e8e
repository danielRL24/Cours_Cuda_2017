#include "Indice2D.h"
#include "Indice1D.h"
#include "cudaTools.h"
#include "RayTracingMath.h"

#include <stdio.h>
#include "IndiceTools_GPU.h"

#include "../length.h"
using namespace gpu;

// Déclaration constante globale
__constant__ float TAB_CM[LENGTH_CM];


/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/
/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

//__device__ void work(Sphere* ptrTabSphere, int n);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * Call once by the host
 */
//__host__ void uploadGPU(Sphere* ptrDevTabSphere)
//    {
//    size_t size = LENGTH_CM * sizeof(Sphere);
//    int offset = 0;
//    HANDLE_ERROR(cudaMemCpyToSymbol(TAB_CM, ptrDevTabSphere, size, offset, cudaMemcpyHostToDevice));
//    }

__global__ void raytracing(uchar4* ptrDevPixels, uint w, uint h, float t, Sphere* ptrDevTabSphere,int tabSphereLength)
    {
    RayTracingMath raytracingMath = RayTracingMath(ptrDevTabSphere, tabSphereLength);

    const int WH = w*h;
    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();

    int i;
    int j;

    int s = TID;
    while(s < WH)
	{
	IndiceTools::toIJ(s, w, &i, &j);
	raytracingMath.colorIJ(&ptrDevPixels[s], i, j, t);
	s += NB_THREAD;
	}
    }

//__global__ void rayTracingCM(...)
//    {
//    // work();
//    }
//
//__global__ void rayTracingSM(...)
//    {
//    // work();
//    }

__device__ void work(uchar4* ptrDevPixels, Sphere* ptrDevTabSphere, int nbSphere, uint w, uint h, float t)
    {
    RayTracingMath raytracingMath = RayTracingMath(ptrDevTabSphere, nbSphere);

    const int WH = w*h;
    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();

    int i;
    int j;

    int s = TID;
    while(s < WH)
	{
	IndiceTools::toIJ(s, w, &i, &j);
	raytracingMath.colorIJ(&ptrDevPixels[s], i, j, t);
	s += NB_THREAD;
	}
    }


/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

