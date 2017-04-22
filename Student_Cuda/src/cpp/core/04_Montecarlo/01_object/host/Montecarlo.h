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
	Montecarlo(const Grid& grid, int m, long n);
	virtual ~Montecarlo();

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/
    public:
	void run();
	double getPiHat();
	long getNx();
	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/
    private:
	dim3 db;
	dim3 dg;
	int M;
	long n;
	int nbThreat;
	long nx;

	size_t sizeOctetGenerator;
	size_t sizeOctetSM;

	double piHat;
	long* ptrDevNx;
	curandState* tabDevGeneratorGM;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
