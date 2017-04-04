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

#include "Slice.h"

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool useSlice(void);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

bool isAlgoPI_OK(float piHat, int n, string title);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool useSlice()
    {

    float piHat = 0;
    int nbSlice = 5;

    // Partie interessante GPGPU
	{
	// Grid cuda
	int mp = Device::getMPCount();
	int coreMP = Device::getCoreCountMP();

	dim3 dg = dim3(mp, 1, 1);  		// disons, a optimiser selon le gpu, peut drastiqument ameliorer ou baisser les performances
	dim3 db = dim3(coreMP, 1, 1);   	// disons, a optimiser selon le gpu, peut drastiqument ameliorer ou baisser les performances
	Grid grid(dg, db);

	Slice slice(grid, &piHat, nbSlice);
	slice.run();
	}

    bool isOk = isAlgoPI_OK(piHat, nbSlice, "PI"); // check result

//    delete[] piHat;

    return isOk;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

bool isAlgoPI_OK(float piHat, int n, string title)
    {
    cout << endl << "[" << title << " running ...]" << endl;
    cout << "n=" << n << endl;

    cout.precision(8);
    cout << "Pi hat  = " << piHat << endl;
    cout << "Pi true = " << PI << endl;

    bool isOk = MathTools::isEquals((float)piHat, (float)PI, (float)1e-6);
    cout << "isOk = " << isOk << endl;

    return isOk;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

