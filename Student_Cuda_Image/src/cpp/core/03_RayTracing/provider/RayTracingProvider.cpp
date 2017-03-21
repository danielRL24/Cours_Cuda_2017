#include "RayTracingProvider.h"
#include "RayTracing.h"

#include "MathTools.h"
#include "Grid.h"

#include "ImageAnimable_GPU.h"

using namespace gpu;

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

/**
 * Override
 */
Animable_I<uchar4>* RayTracingProvider::createAnimable()
    {
    // Animation;
    float dt = 2.f * PI_FLOAT / 1000;

    // Sphere
    int nbSphere = 10;

    // Dimension
    int dw = 800;
    int dh = 800;

    // Grid Cuda
    int mp = Device::getMPCount();
    int coreMP = Device::getCoreCountMP();

    dim3 dg = dim3(24, 1, 1);  		// disons, a optimiser selon le gpu, peut drastiqument ameliorer ou baisser les performances
    dim3 db = dim3(1024, 1, 1);   	// disons, a optimiser selon le gpu, peut drastiqument ameliorer ou baisser les performances
    Grid grid(dg, db);

    return new RayTracing(grid, dw, dh, dt, nbSphere);
    }

/**
 * Override
 */
Image_I* RayTracingProvider::createImageGL(void)
    {
    ColorRGB_01 colorTexte(0, 0, 0); // Black
    return new ImageAnimable_RGBA_uchar4(createAnimable(), colorTexte);
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
