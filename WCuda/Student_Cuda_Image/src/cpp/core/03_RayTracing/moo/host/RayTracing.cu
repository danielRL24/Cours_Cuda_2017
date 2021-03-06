#include <iostream>
#include <assert.h>

#include "RayTracing.h"
#include "SphereCreator.h"
#include "Sphere.h"
#include "length.h"

#include "Device.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern __host__ void uploadToCM(Sphere* ptrTabSphere);
extern __global__ void raytracingGM(uchar4* ptrDevPixels, uint w, uint h, float t, Sphere* ptrDevTabSphere,int tabSphereLength);
extern __global__ void raytracingCM(uchar4* ptrDevPixels, uint w, uint h, float t);
extern __global__ void raytracingSM(uchar4* ptrDevPixels, uint w, uint h, float t, Sphere* ptrDevTabSphere, int tabSphereLength);

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

/*-------------------------*\
 |*	Constructeur	    *|
 \*-------------------------*/

RayTracing::RayTracing(const Grid& grid, uint w, uint h, float dt, int nbSphere) :
	Animable_I<uchar4>(grid, w, h, "RayTracing_Cuda_RGBA_uchar4")
    {
    this->dt = dt;
    this->t = 0;
    this->sizeOctet = nbSphere * sizeof(Sphere);
    this->nbSphere = nbSphere;

    SphereCreator sphereCreator(nbSphere, w, h); // sur la pile
    Sphere* ptrTabSphere = sphereCreator.getTabSphere();
    // transfert to GM
    toGM(ptrTabSphere);

    // Appelle le service d'upload coté device
    uploadToCM(ptrTabSphere);
    }

RayTracing::~RayTracing()
    {
    //MM (device free)
	{
	Device::free(ptrDevTabSphere);

	Device::lastCudaError("RayTracing MM (end deallocation)"); // temp debug, facultatif
	}
    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/

void RayTracing::toGM(Sphere* ptrTabSphere)
    {
    // MM (malloc Device)
	{
	Device::malloc(&ptrDevTabSphere, sizeOctet);
	Device::memclear(ptrDevTabSphere, sizeOctet);
	}

    // MM (copy Host->Device)
	{
	Device::memcpyHToD(ptrDevTabSphere, ptrTabSphere, sizeOctet);
	}

    Device::lastCudaError("RayTracing MM (end allocation)"); // temp debug, facultatif
    }

/**
 * Override
 * Call periodicly by the API
 *
 * Note : domaineMath pas use car pas zoomable
 */
void RayTracing::process(uchar4* ptrDevPixels, uint w, uint h, const DomaineMath& domaineMath)
    {
    Device::lastCudaError("raytracing (before)"); // facultatif, for debug only, remove for release

    static int i = 0;

    if (i % 3 == 0)
	{
	raytracingGM<<<dg,db>>>(ptrDevPixels,w,h,t,ptrDevTabSphere,nbSphere);
	}
    else if (i % 3 == 1)
	{
	raytracingCM<<<dg,db>>>(ptrDevPixels, w, h, t);
	}
    else if (i % 3 == 2)
	{
	raytracingSM<<<dg,db, sizeOctet>>>(ptrDevPixels, w, h, t, ptrDevTabSphere, nbSphere);
	}
    i++;

    Device::lastCudaError("raytracing (after)"); // facultatif, for debug only, remove for release
    }

/**
 * Override
 * Call periodicly by the API
 */
void RayTracing::animationStep()
{
t += dt;
}
/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

