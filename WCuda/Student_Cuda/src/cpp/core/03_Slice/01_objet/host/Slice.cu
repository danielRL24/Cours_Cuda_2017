#include <iostream>

#include "Device.h"
#include "Slice.h"

#include <curand_kernel.h>

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

extern __global__ void slice(float* ptrDevGM, int nbSlice);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Constructeur			*|
 \*-------------------------------------*/

Slice::Slice(const Grid& grid, float* ptrResult, int nbSlice) :
	nbSlice(nbSlice), ptrResult(ptrResult)
    {
    this->sizeOctetGM = sizeof(float);
    this->sizeOctetSM = sizeof(float) * grid.db.x;

    // MM
	{

	// MM (malloc Device)
	    {
	    Device::malloc(&ptrDevResult, sizeOctetGM);
	    Device::memclear(ptrDevResult, sizeOctetGM);
	    }

	Device::lastCudaError("Slice MM (end allocation)"); // temp debug, facultatif
	}

    // Grid
	{
	this->dg = grid.dg;
	this->db = grid.db;
	}
    }

Slice::~Slice()
    {
    //MM (device free)
	{
	Device::free(ptrDevResult);

	Device::lastCudaError("Slice  MM (end deallocation)"); // temp debug, facultatif
	}
    }

/*--------------------------------------*\
 |*		Methode			*|
 \*-------------------------------------*/

void Slice::run()
    {
    Device::lastCudaError("Slice  (before)"); // temp debug
    slice<<<dg,db, sizeOctetSM >>>(ptrDevResult, nbSlice); // assynchrone
    Device::lastCudaError("Slice  (after)"); // temp debug

    // Debug, facultatif (voir addVector_device.cu)
//     Device::synchronize(); // Temp,debug, only for printf in  GPU

    // MM (Device -> Host)
	{
	Device::memcpyDToH(ptrResult, ptrDevResult, sizeOctetGM); // barriere synchronisation implicite
	}

    }


/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

