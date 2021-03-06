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

	    float distance = 0;
	    float closestDistance = 10000;
	    float dz = 0;
	    float brightness = 0;
	    float hue = 0;
	    float hCarre = 0;

	    Sphere closestSphere = ptrTabSphere[0];

	    int k = 0;
	    for (k = 0; k < tabSphereLength; k++)
		{
		Sphere sphereK = ptrTabSphere[k];
		hCarre = sphereK.hCarre(xySol);

		// isEnDessous remplace une condition
		int isEnDessous = (int)sphereK.isEnDessous(hCarre);
		dz = sphereK.dz(hCarre);
		distance = sphereK.distance(dz);
		if(distance * isEnDessous < closestDistance * isEnDessous)
		    {
		    closestSphere = sphereK;
		    closestDistance = distance;
		    }
		}

		hCarre = closestSphere.hCarre(xySol);
		dz = closestSphere.dz(hCarre);
		brightness = closestSphere.brightness(dz);
		hue = closestSphere.hue(t);
		ColorTools::HSB_TO_RVB(hue, 1, brightness, ptrColor);
	    }

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	Sphere* ptrTabSphere;
	int tabSphereLength;

    }
;

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
