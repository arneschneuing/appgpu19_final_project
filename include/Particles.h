#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>
#include <algorithm>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;
    
    
    
};

/** allocate particle arrays */
void particle_allocate(struct parameters*, struct particles*, int);

/** deallocate */
void particle_deallocate(struct particles*);

/** launcher for gpu particle mover */
int mover_PC_gpu_launch(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/* launcher for GPU version of the P2G interpolation */
int interpP2G_gpu_launch(struct particles*, struct interpDensSpecies*, struct grid*, struct parameters*);

/**
* Create batches of particles
* 
* @param param Structure containing the simulation parameters
* @param part Particle structure containing all particles (of one species)
* @param part_batches Array of particle structures storing the batches after creation
* @return number of batches
*/
int particle_batch_create(struct parameters* param, struct particles* part, struct particles** part_batches);

/**
* Deallocate particle batches
* 
* @param part_batches
* @param nob Number of batches
*/
void particle_batch_deallocate(struct particles* part_batches, int nob);

#endif
