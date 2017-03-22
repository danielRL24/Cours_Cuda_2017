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

//	    float2 xySol;
//	    xySol.x = j;
//	    xySol.y = i;
//
//	    float distance = 0;
//	    float closestDistance = 0;
//	    float dz = 0;
//	    float brightness = 0;
//	    float hue = 0;
//	    float hCarre = 0;
//	    int isEnDessous = 0;
//
//	    Sphere sphereK;
//
//	    int k = 0;
//	    for(k = 0; k < tabSphereLength; k++)
//		{
//		Sphere sphereK = ptrTabSphere[k];
//		hCarre = sphereK.hCarre(xySol);
//
//		// Remplace le IF
//		// Si false => isEnDessous = 0
//		// Si true  => isEnDessous = 1
//		isEnDessous = (int) sphereK.isEnDessous(hCarre);
//		dz = sphereK.dz(hCarre);
//		brightness = sphereK.brightness(dz) * isEnDessous;
//		hue = (sphereK.getHueStart() + sphereK.hue(t)) * isEnDessous;
//		ColorTools::HSB_TO_RVB(hue, 1, brightness, ptrColor);
//
//		// Condition de sortie si une sphere est en-dessus
//		k += tabSphereLength * isEnDessous;
//
//		}

	    float2 xySol;
	    xySol.x = j;
	    xySol.y = i;

//	    float distance = 0;
//	    float closestDistance = 0;
	    float dz = 0;
	    float brightness = 0;
	    float hue = 0;
	    float hCarre = 0;
	    int isEnDessous = 0;

//	    Sphere sphereK;

	    int k = 0;
	    for (k = 0; k < tabSphereLength; k++)
		{
		Sphere sphereK = ptrTabSphere[k];
		hCarre = sphereK.hCarre(xySol);

		// Remplace le IF
		// Si false => isEnDessous = 0
		// Si true  => isEnDessous = 1
		isEnDessous = (int) sphereK.isEnDessous(hCarre);
		dz = sphereK.dz(hCarre);
		brightness = sphereK.brightness(dz) * isEnDessous;
		hue = (sphereK.getHueStart() + sphereK.hue(t)) * isEnDessous;
		ColorTools::HSB_TO_RVB(hue, 1, brightness, ptrColor);

		// Condition de sortie si une sphere est en-dessus
		k += tabSphereLength * isEnDessous;
		}

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
