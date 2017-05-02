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
extern __global__ void montecarlo(long* ptrDevNx, curandState* tabDevGenerator, long n, float m);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

Montecarlo::Montecarlo(const Grid& grid, int m, long n) :
	M(m), n(n)
    {
	this->db = grid.db;
	this->dg = grid.dg;
	this->nbThreat = grid.threadCounts();
	this->nx = 0;
	this->piHat = 0;
	this->sizeOctetGenerator = nbThreat * sizeof(curandState);
	this->sizeOctetSM = sizeof(long) * db.x;

	Device::malloc(&ptrDevNx, sizeof(long));
	Device::memclear(ptrDevNx, sizeof(long));

	Device::malloc(&tabDevGeneratorGM, sizeOctetGenerator);
	Device::memclear(tabDevGeneratorGM, sizeOctetGenerator);
    }

Montecarlo::~Montecarlo()
    {
    Device::free(ptrDevNx);
    Device::free(tabDevGeneratorGM);
    }


void Montecarlo::run()
    {
    int deviceID = Device::getDeviceId();
    Device::lastCudaError("Montecarlo (before)"); // temp debug
    setup_kernel_rand<<<dg, db>>>(tabDevGeneratorGM, deviceID);
    montecarlo<<<dg, db, sizeOctetSM>>>(ptrDevNx, tabDevGeneratorGM, n, M);
    Device::lastCudaError("Montecarlo (after)"); // temp debug

    Device::memcpyDToH(&nx, ptrDevNx, sizeof(long)); // barriere synchronisation implicite

    piHat = (double)M * ((double)nx/(double)n);
    }

double Montecarlo::getPiHat()
    {
    return piHat;
    }

long Montecarlo::getNx()
    {
    return nx;
    }
/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

