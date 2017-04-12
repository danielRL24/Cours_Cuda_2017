#include <iostream>
#include "VectorTools.h"
#include "Grid.h"
#include "Device.h"
#include "Chrono.h"
#include "MathTools.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

#include "Montecarlo.h"

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool useMontecarlo(void);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool useMontecarlo()
    {

    float piHat = 0;
    int m = 5;		// Taille de la cible
    long n = 1000000; 	// Nombre de flechettes

    // Partie interessante GPGPU
	{
	// Grid cuda
	int mp = Device::getMPCount();
	int coreMP = Device::getCoreCountMP();

	dim3 dg = dim3(mp, 1, 1);
	dim3 db = dim3(coreMP, 1, 1);
	Grid grid(dg, db);

	Montecarlo montecarlo(grid, &piHat, m, n);
	montecarlo.run();
	}

	cout << endl << "[Montecarlo running ...]" << endl;
	cout << "n=" << n << " & m=" << m << endl;

	cout.precision(8);
	cout << "Pi hat  = " << piHat << endl;
	cout << "Pi true = " << PI << endl;

	bool isOk = MathTools::isEquals((float)piHat, (float)PI, (float)1e-6);
	cout << "Montecarlo : isOk = " << isOk << endl;

	return isOk;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

