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
	    xySol.x = i;
	    xySol.y = j;
	    float hCarre = 0;
	    float brightness = 0;
	    float hue = ptrTabSphere[0].getHueStart();
	    float dz = 0;

	    ptrColor->x = 0;
	    ptrColor->y = 0;
	    ptrColor->z = 0;

	    ptrColor->w = 255;

	    int k = 0;
	    for (k = 0; k < tabSphereLength; k++)
		{
		hCarre = ptrTabSphere[k].hCarre(xySol);
		if (ptrTabSphere[k].isEnDessous(hCarre))
		    {
		    dz = ptrTabSphere[k].dz(hCarre);
//		    float distance = ptrTabSphere[k].distance(dz);
		    brightness = ptrTabSphere[k].brightness(dz);
		    hue = ptrTabSphere[k].hue(t);
		    ColorTools::HSB_TO_RVB(hue, 1, brightness, ptrColor);

		    ptrColor->x = 155;
		    ptrColor->y = 155;
		    ptrColor->z = 155;

		    ptrColor->w = 255;
		    break;
		    }
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