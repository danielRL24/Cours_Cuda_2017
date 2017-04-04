#include <iostream>

#include "Device.h"
#include "Slice.h"

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
    this->sizeOctet = sizeof(float); // octet

    // MM
	{

	// MM (malloc Device)
	    {
	    Device::malloc(&ptrDevResult, sizeOctet);
	    Device::memclear(ptrDevResult, sizeOctet);
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
    size_t sizeOfSM = sizeOctet * db.x * db.y * db.z;
    Device::lastCudaError("Slice  (before)"); // temp debug
    slice<<<dg,db, sizeOctet >>>(ptrDevResult, nbSlice); // assynchrone
    Device::lastCudaError("Slice  (after)"); // temp debug

    // Debug, facultatif (voir addVector_device.cu)
//     Device::synchronize(); // Temp,debug, only for printf in  GPU

    // MM (Device -> Host)
	{
	Device::memcpyDToH(ptrResult, ptrDevResult, sizeOctet); // barriere synchronisation implicite
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

