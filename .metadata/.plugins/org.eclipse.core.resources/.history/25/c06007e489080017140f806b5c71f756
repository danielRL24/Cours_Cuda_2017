#pragma once

#include <math.h>
#include "MathTools.h"

#include "Calibreur_GPU.h"
#include "ColorTools_GPU.h"
using namespace gpu;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class MandelbrotMath
    {

	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	__device__ MandelbrotMath(uint n) :
		calibreur(Interval<float>(1, n), Interval<float>(0, 1))
	    {
	    }

	__device__ virtual ~MandelbrotMath()
	    {
	    // rien
	    }

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

    public:

	__device__ void colorXY(uchar4* ptrColor, float x, float y, int t)
	    {
	    int z = f(x, y, t);

	    if (z >= t)
		{
		ptrColor->x = 0;
		ptrColor->y = 0;
		ptrColor->z = 0;
		ptrColor->w = 255;
		}
	    else
		{
		float hue01 = z;
		calibreur.calibrer(hue01);

		ColorTools::HSB_TO_RVB(hue01, ptrColor); // update color

		ptrColor->w = 255; // opaque
		}

	    }

    private:

	__device__ int f(float x, float y, int t)
	    {
	    float a = 0;
	    float b = 0;
	    int k = 0;

	    while ((a * a + b * b) < 4 && k < t)
		{
		float aCopy = a;
		a = ((a * a) - (b * b)) + x;
		b = (2 * aCopy * b) + y;
		k++;
		}

	    return k;
	    }

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Tools
	Calibreur<float> calibreur;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
