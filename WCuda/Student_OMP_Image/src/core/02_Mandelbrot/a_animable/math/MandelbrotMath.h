#pragma once

#include <math.h>
#include "MathTools.h"

#include "Calibreur_CPU.h"
#include "ColorTools_CPU.h"
using namespace cpu;

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

	MandelbrotMath(uint n) :
		calibreur(Interval<float>(1, n), Interval<float>(0, 1))
	    {
	    }

	virtual ~MandelbrotMath()
	    {
	    // rien
	    }

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

    public:

	void colorXY(uchar4* ptrColor, float x, float y, int t)
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

	int f(float x, float y, int t)
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
