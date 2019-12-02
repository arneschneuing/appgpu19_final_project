#ifndef PARTICLES_AUX_H
#define PARTICLES_AUX_H

#include "Particles.h"

struct particles_aux {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** densities carried nop,2,2,2*/
    FPpart (*rho_p)[2][2][2];
    FPpart (*Jx)[2][2][2]; FPpart (*Jy)[2][2][2]; FPpart (*Jz)[2][2][2];
    FPpart (*pxx)[2][2][2]; FPpart (*pxy)[2][2][2]; FPpart (*pxz)[2][2][2];
    FPpart (*pyy)[2][2][2]; FPpart (*pyz)[2][2][2]; FPpart (*pzz)[2][2][2];
    
    
    /** cell index: ix, iy, iz */
    int* ix_p; int* iy_p; int* iz_p;
    
};

/** allocate particle arrays */
void particle_aux_allocate(struct particles*, struct particles_aux*, int);

void particle_aux_deallocate(struct particles_aux*);

#endif

