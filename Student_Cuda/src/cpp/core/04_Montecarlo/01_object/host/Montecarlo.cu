#include "Montecarlo.h"
#include <curand_kernel.h>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

Montecarlo::Montecarlo(const Grid& grid, float* ptrPiHat, float m, long n) :
    ptrPiHat(ptrPiHat), m(m), n(n)
    {
	this->db = grid.db;
	this->dg = grid.dg;
	this->nbThreat = grid.threadCounts();
	this->sizeOctetGenerator = nbThreat * sizeof(curandState);
	this->sizeOctetSM = sizeof(float) * db.x * db.z * db.x;

	Device::malloc(&ptrNx, sizeof(int));
	Device::memclear(ptrNx, sizeof(int))

	Device::malloc(&ptrDevNx, sizeof(int));
	Device::memclear(ptrDevNx, sizeof(int));

	Device::malloc(&tabDevGeneratorGM, sizeOctetGenerator);
	Device::memclear(tabDevGeneratorGM, sizeOctetGenerator);
    }

Montecarlo::~Montecarlo()
    {
    Device::free(ptrNx);
    Device::free(ptrDevNx);
    Device::free(tabDevGeneratorGM);
    }


void Montecarlo::run()
    {
    int deviceID
    Device::lastCudaError("Montecarlo (before)"); // temp debug
    setup_kernel_rand<<<dg, db>>>(tabDevGeneratorGM, deviceID);
    montecalo<<<dg, db, sizeOctetSM>>>(ptrDevNx, tabDevGeneratorGM, n, m);
    Device::lastCudaError("Montecarlo (after)"); // temp debug

    // Debug, facultatif (voir addVector_device.cu)
//     Device::synchronize(); // Temp,debug, only for printf in  GPU

    Device::memcpyDToH(ptrNx, ptrDevNx, sizeOctetGM); // barriere synchronisation implicite

    }
/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

