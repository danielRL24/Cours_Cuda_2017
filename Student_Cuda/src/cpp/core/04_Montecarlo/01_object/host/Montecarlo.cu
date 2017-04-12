#include <iostream>

#include "Montecarlo.h"
#include "Device.h"
#include <curand_kernel.h>

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

extern __global__ void setup_kernel_rand(curandState* tabDevGenerator, int deviceID);
extern __global__ void montecarlo(int* ptrDevNx, curandState* tabDevGenerator, long n, float m);

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
	this->nx = 0;
	this->sizeOctetGenerator = nbThreat * sizeof(curandState);
	this->sizeOctetSM = sizeof(int) * db.x;

	Device::malloc(&ptrDevNx, sizeof(int));
	Device::memclear(ptrDevNx, sizeof(int));

	Device::malloc(&tabDevGeneratorGM, sizeOctetGenerator);
	Device::memclear(tabDevGeneratorGM, sizeOctetGenerator);


	int deviceID = Device::getDeviceId();


	setup_kernel_rand<<<dg, db>>>(tabDevGeneratorGM, deviceID);
    }

Montecarlo::~Montecarlo()
    {
    Device::free(ptrDevNx);
    Device::free(tabDevGeneratorGM);
    }


void Montecarlo::run()
    {
    Device::lastCudaError("Montecarlo (before)"); // temp debug
    montecarlo<<<dg, db, sizeOctetSM>>>(ptrDevNx, tabDevGeneratorGM, n, m);
    Device::lastCudaError("Montecarlo (after)"); // temp debug

    // Debug, facultatif (voir addVector_device.cu)
//     Device::synchronize(); // Temp,debug, only for printf in  GPU

    Device::memcpyDToH(&nx, ptrDevNx, sizeof(int)); // barriere synchronisation implicite

    *ptrPiHat = ((float)nx * (float)m)/(float)n;
    }
/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
