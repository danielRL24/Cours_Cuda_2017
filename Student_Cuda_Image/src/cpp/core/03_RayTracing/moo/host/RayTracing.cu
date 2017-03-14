#include "RayTracing.h"
#include "SphereCreator.h"
#include "Sphere.h"

#include <iostream>
#include <assert.h>

#include "Device.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern __global__ void raytracing(Sphere* ptrDevTabSphere, uint w, uint h, float t);

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

    ShereCreator sphereCreator(nbSphere, w, h); // sur la pile
    Sphere* ptrTabSphere = sphereCreator.getTab();
    // transfert to GM
    toGM(ptrTabSphere);
    // transfert to CM
//    toCM(ptrTabSphere);

    // Grid
	{
	this->dg = grid.dg;
	this->db = grid.db;
	}
    }

RayTracing::~RayTracing()
    {
    //MM (device free)
    	{
    	Device::free(ptrDevTabSphere);
    	Device::free(ptrDevResult);

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
	Device::malloc(ptrDevTabSphere, sizeOctet);
	Device::malloc(ptrDevResult, sizeOctet);
	Device::memclear(ptrDevTabSphere, sizeOctet);
	Device::memclear(ptrDevResult, sizeOctet);
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
void RayTracing::process(Sphere* ptrDevTabSphere, uint w, uint h, const DomaineMath& domaineMath)
    {
    Device::lastCudaError("raytracing (before)"); // facultatif, for debug only, remove for release
    raytracing<<<dg,db>>>(ptrDevTabSphere,w,h,t); // Drivers nVidia s'occupe de transformer les types simples
    Device::lastCudaError("raytracing (after)"); // facultatif, for debug only, remove for release

    // MM (Device -> Host)
//	{
//	Device::memcpyDToH(ptrW, ptrDevW, sizeOctet); // barriere synchronisation implicite
//	}
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

