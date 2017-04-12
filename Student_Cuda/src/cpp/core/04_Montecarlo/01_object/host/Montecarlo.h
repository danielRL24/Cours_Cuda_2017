#pragma once

#include "cudaTools.h"
#include "Grid.h"
#include <curand_kernel.h>
/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class Montecarlo
    {
    public:
	Montecarlo(const Grid& grid, float* ptrPiHat, float m, long n);
	virtual ~Montecarlo();

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/
    public:
	void run();
	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/
    private:
	dim3 db;
	dim3 dg;
	float m;
	float n;
	int nbThreat;
	int nx;

	size_t sizeOctetGenerator;
	size_t sizeOctetSM;

	float* ptrPiHat;
	int* ptrDevNx;
	curandState* tabDevGeneratorGM;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/