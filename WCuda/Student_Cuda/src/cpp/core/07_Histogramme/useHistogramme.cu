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

#include "Histogramme.h"

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool useHistrogramme(void);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool useHistogramme()
    {
    int* ptrTabResult;

    // Partie interessante GPGPU
	{
	dim3 dg = dim3(32, 1, 1);
	dim3 db = dim3(64, 1, 1);
	Grid grid(dg, db);

	Histogramme histogramme(grid);
	ptrTabResult = histogramme.run();
	}

	bool isOk = true;

	cout << endl << "[Histrogramme running ...]" << endl;
	for(int i=0; i<256; i++)
	    {
	    if(i > 0)
		{
		isOk &= (ptrTabResult[i]==ptrTabResult[i-1]+1);
		}
	    cout << ptrTabResult[i] << ", ";
	    }

	return true;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

