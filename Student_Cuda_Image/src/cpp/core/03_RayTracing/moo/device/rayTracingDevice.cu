#include "cudaTools.h"
#include "Indice2D.h"

#include "IndiceTools_GPU.h"
#include "Device.h"

#include "RayTracingMath.h"

#include "../length.h"

using namespace gpu;
// Déclaration constante globale
__constant__ Sphere TAB_CM[LENGTH_CM];


/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/
/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__device__ void work(uchar4* ptrDevPixels, Sphere* ptrDevTabSphere, int nbSphere, uint w, uint h, float t, const int TID, const int NB_THREAD)
    {
    RayTracingMath raytracingMath = RayTracingMath(ptrDevTabSphere, nbSphere);

    const int WH = w*h;

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

__global__ void raytracingGM(uchar4* ptrDevPixels, uint w, uint h, float t, Sphere* ptrDevTabSphere,int tabSphereLength)
    {
    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();

    work(ptrDevPixels, ptrDevTabSphere, tabSphereLength, w, h, t, TID, NB_THREAD);
    }

__global__ void raytracingCM(uchar4* ptrDevPixels, uint w, uint h, float t)
    {
    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();

    work(ptrDevPixels, TAB_CM, LENGTH_CM, w, h, t, TID, NB_THREAD);
    }

__global__ void raytracingSM(uchar4* ptrDevPixels, uint w, uint h, float t, Sphere* ptrDevTabSphereGM, int tabSphereLength)
    {
    const int TID_LOCAL = Indice2D::tidLocal();
    const int NB_THREAD_LOCAL = Indice2D::nbThreadLocal();

    extern __shared__ Sphere ptrDevTabSphereSM[];

    int s = TID_LOCAL;
    while(s < tabSphereLength)
	{
	ptrDevTabSphereSM[s] = ptrDevTabSphereGM[s];
	s++;
	}

    work(ptrDevPixels, ptrDevTabSphereSM, tabSphereLength, w, h, t, TID_LOCAL, NB_THREAD_LOCAL);
    }

/**
 * Call once by the host
 */
__host__ void uploadGPU(Sphere* ptrDevTabSphere)
    {
    size_t size = LENGTH_CM * sizeof(Sphere);
    int offset = 0;
    Device::memcpyToCM(TAB_CM, ptrDevTabSphere, size);
//    HANDLE_ERROR(cudaMemCpyToSymbol(TAB_CM, ptrDevTabSphere, size, offset, cudaMemcpyHostToDevice));
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

