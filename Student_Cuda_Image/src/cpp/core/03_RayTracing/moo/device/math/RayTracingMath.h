#pragma once

#include <math.h>
#include "MathTools.h"
#include "Sphere.h"

#include "ColorTools_GPU.h"
using namespace gpu;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class RayTracingMath
    {

	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	__device__ RayTracingMath(Sphere* ptrDevTabSphere, int tabSphereLength)
	    {
	    this->ptrTabSphere = ptrDevTabSphere;
	    this->tabSphereLength = tabSphereLength;
	    }

	// constructeur copie automatique car pas pointeur dans VagueMath

	__device__
	       virtual ~RayTracingMath()
	    {
	    // rien
	    }

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

    public:

	__device__
	void colorIJ(uchar4* ptrColor, int i, int j, float t)
	    {

	    float2 xySol;
	    xySol.x = j;
	    xySol.y = i;

	    ptrColor->x = 0;
	    ptrColor->y = 0;
	    ptrColor->z = 0;
	    ptrColor->w = 255;

	    int k = 0;
	    for(k = 0; k < tabSphereLength; k++)
		{
		Sphere sphereK = ptrTabSphere[k];
		float hCarre = sphereK.hCarre(xySol);

		int color = (int) sphereK.isEnDessous(hCarre);

		float dz = sphereK.dz(hCarre);
		float distance = sphereK.distance(dz) * color;
		float brightness = sphereK.brightness(dz) * color;
		float hue = sphereK.hue(t) * color;
		ColorTools::HSB_TO_RVB(hue, 1, brightness, ptrColor);

		k += tabSphereLength * color;
		}
	    }

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	Sphere* ptrTabSphere;
	int tabSphereLength;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
