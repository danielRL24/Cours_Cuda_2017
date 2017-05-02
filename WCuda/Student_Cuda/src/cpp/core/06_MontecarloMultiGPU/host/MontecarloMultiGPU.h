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

class MontecarloMultiGPU
    {
    public:
	MontecarloMultiGPU(const Grid& grid, int m, long n);
	virtual ~MontecarloMultiGPU();


	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/
    public:
	void run();
	double getPiHat();
	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/
    private:
	Grid grid;
	int M;
	long n;

	double piHat;
    };
/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
