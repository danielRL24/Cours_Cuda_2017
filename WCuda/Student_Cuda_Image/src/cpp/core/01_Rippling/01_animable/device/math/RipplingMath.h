#pragma once

#include <math.h>
#include "MathTools.h"

#include "ColorTools_GPU.h"
using namespace gpu;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class RipplingMath
    {

	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	__device__ RipplingMath(int w, int h)
	    {
	    this->dim2 = w / 2.0f;
	    }

	// constructeur copie automatique car pas pointeur dans VagueMath

	__device__
	   virtual ~RipplingMath()
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
	    uchar levelGris;

	    f(&levelGris, i, j, t); // update levelGris

	    ptrColor->x = levelGris;
	    ptrColor->y = levelGris;
	    ptrColor->z = levelGris;

	    ptrColor->w = 255; // opaque
	    }

    private:

	__device__
	void f(uchar* ptrLevelGris, int i, int j, float t)
	    {
	    // cf fonction math pdf
	    // use focntion dij ci-dessous

	    // *ptrLevelGris = 128;
	    float rdij = dij(i, j);
	    rdij = rdij/10.0f;

	    float nominateur = cosf(rdij-t/7.0f);
	    float denominateur = rdij+1.0f;

	    *ptrLevelGris = 128.0f + 127.0f * (nominateur / denominateur);

	    // Note
	    //		Si code OMP focntionnel:
	    // 			Step1 : Delete le contenur de ce fichier (si!),
	    // 			Step2 : Copie-past le contenu de RipplingMath.h de omp,
	    // 			Step3 : Ajouter __device__  devant methode et constructeur!
	    }

	__device__
	float  dij(int i, int j)
	    {
	    // cf fonction math pdf
	    float fi = i - this->dim2;
	    float fj = j - this->dim2;

	    return sqrtf((fi * fi) + (fj * fj));
	    }

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Tools
	float dim2;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
