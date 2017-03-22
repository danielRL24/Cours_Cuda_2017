#include "RayTracing.h"
#include "SphereCreator.h"
#include "Sphere.h"

#include <iostream>
#include <assert.h>

#include "Device.h"
#include "../length.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern __global__ void raytracing(uchar4* ptrDevPixels, uint w, uint h, float t, Sphere* ptrDevTabSphere, int tabSphereLength);

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
    // transfert to CM
    // toCM(ptrTabSphere);

    // Appelle le service d'upload coté device
//    uploadGPU(ptrTabSphere);
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

//void RayTracing::toCM(Sphere* ptrTabSphere)
//    {
//
//    }

/**
 * Override
 * Call periodicly by the API
 *
 * Note : domaineMath pas use car pas zoomable
 */
void RayTracing::process(uchar4* ptrDevPixels, uint w, uint h, const DomaineMath& domaineMath)
    {
    Device::lastCudaError("raytracing (before)"); // facultatif, for debug only, remove for release
    raytracing<<<dg,db>>>(ptrDevPixels,w,h,t,ptrDevTabSphere,nbSphere); // Drivers nVidia s'occupe de transformer les types simples
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

