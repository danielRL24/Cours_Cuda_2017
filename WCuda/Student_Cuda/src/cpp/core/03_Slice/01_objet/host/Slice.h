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

class Slice
    {
    public:
	Slice(const Grid& grid, float* ptrResult, int nbSlice);
	virtual ~Slice();

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

    public:
	void run();

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/
    private:
	// Inputs
	dim3 dg;
	dim3 db;
	int nbSlice;

	// Outputs
	float* ptrResult;

	// Tools
	float* ptrDevResult;
	size_t sizeOctetGM;
	size_t sizeOctetSM;

    };
/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
