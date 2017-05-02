#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "Device.h"
#include "cudaTools.h"

#include "RipplingProvider.h"
#include "MandelbrotProvider.h"
#include "RayTracingProvider.h"

#include "Settings_GPU.h"
#include "Viewer_GPU.h"
using namespace gpu;

using std::cout;
using std::endl;
using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainImage(Settings& settings);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainImage(Settings& settings)
    {
    cout << "\n[Image] mode" << endl;

    GLUTImageViewers::init(settings.getArgc(), settings.getArgv()); //only once

    // ImageOption : (boolean,boolean) : (isSelection ,isAnimation,isOverlay,isShowHelp)
    ImageOption zoomable(true,true,true,true);
    ImageOption nozoomable(false,true,true,true);

    Viewer<RipplingProvider> rippling(nozoomable, 25, 25); // imageOption px py
    rippling.setSize(400, 400);
    rippling.setPosition(840, 0);
    Viewer<MandelbrotProvider> mandelbrot(zoomable, 25, 25);
    mandelbrot.setSize(400, 400);
    mandelbrot.setPosition(420, 0);
    Viewer<RayTracingProvider> raytracing(nozoomable, 25, 25);
    raytracing.setSize(400, 400);
    raytracing.setPosition(0, 0);

    // Common
    GLUTImageViewers::runALL(); // Bloquant, Tant qu'une fenetre est ouverte

    cout << "\n[Image] end" << endl;

    return EXIT_SUCCESS;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

