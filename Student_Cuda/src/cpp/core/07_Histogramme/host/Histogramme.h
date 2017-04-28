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

class Histogramme
    {
    public:
	Histogramme(const Grid& grid);
	virtual ~Histogramme();

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/
    public:
	int* run();

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/
    private:
	dim3 dg, db;
	int tabSize;

	int* ptrTabData;
	int* ptrTabResult;
	int* ptrDevTabResult;
	int* ptrDevTabData;

	size_t sizeOctetGM;
	size_t sizeOctetSM;
    };
/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
