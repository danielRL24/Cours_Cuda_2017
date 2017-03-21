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

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

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
	IndiceTools::toIJ(s, w, &j, &j);
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

