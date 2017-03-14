#include "Indice2D.h"
#include "Indice1D.h"
#include "cudaTools.h"
#include "RayTracingMath.h"

#include <stdio.h>
#include "IndiceTools_GPU.h"
using namespace gpu;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void raytracing(Sphere* ptrDevTabSpheres,uint w, uint h,float t);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void raytracing(Sphere* ptrDevTabSpheres, uint w, uint h, float t)
    {
    RayTracing raytracing = RayTracing(w, h);

    const int WH = w*h;
    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();

    int i;
    int j;

    int s = TID;
    while(s < n)
	{
	IndiceTools::toIJ(s, w, &j, &j);
	raytracingMath.colorIJ(&ptrDevTabSpheres[s], i, j, t);
	s += NB_THREAD;
	}
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

