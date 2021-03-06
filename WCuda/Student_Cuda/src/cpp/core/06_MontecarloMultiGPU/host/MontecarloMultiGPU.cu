#include <iostream>

#include "MontecarloMultiGPU.h"
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

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

MontecarloMultiGPU::MontecarloMultiGPU(const Grid& grid, int m, long n) :
	 grid(grid), M(m), n(n)
    {
	this->piHat = 0;
    }

MontecarloMultiGPU::~MontecarloMultiGPU()
    {

    }

void MontecarloMultiGPU::run()
    {
	int nbDevice = Device::getDeviceCount();
	int nx = 0;

# pragma omp parallel for reduction(+ : nx)
	for(int deviceID = 0; deviceID < nbDevice; deviceID++)
	    {
		Device::setDevice(deviceID);
		Montecarlo montecarlo (grid, M, n/nbDevice);
		montecarlo.run();
		nx = montecarlo.getNx();
	    }

	piHat = (double)M * ((double)nx/(double)n);
    }

double MontecarloMultiGPU::getPiHat()
    {
	return piHat;
    }
/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

