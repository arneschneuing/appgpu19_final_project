#include "Particles_aux.h"

/** allocate particle arrays */
void particle_aux_allocate(struct particles* part, struct particles_aux* part_aux, int is)
{
    
    // set species ID
    part_aux->species_ID = is;
    // number of particles
    part_aux->nop = part->nop;
    // maximum number of particles
    part_aux->npmax = part->npmax;
    
    long npmax = part->npmax;
    
    // allocate densities brought by each particle
    part_aux->rho_p  = new FPpart[part->npmax][2][2][2];
    part_aux->Jx  = new FPpart[part->npmax][2][2][2];
    part_aux->Jy  = new FPpart[part->npmax][2][2][2];
    part_aux->Jz  = new FPpart[part->npmax][2][2][2];
    part_aux->pxx  = new FPpart[part->npmax][2][2][2];
    part_aux->pxy  = new FPpart[part->npmax][2][2][2];
    part_aux->pxz  = new FPpart[part->npmax][2][2][2];
    part_aux->pyy  = new FPpart[part->npmax][2][2][2];
    part_aux->pyz  = new FPpart[part->npmax][2][2][2];
    part_aux->pzz  = new FPpart[part->npmax][2][2][2];
    
    // cell index
    part_aux->ix_p = new int[part->npmax];
    part_aux->iy_p = new int[part->npmax];
    part_aux->iz_p = new int[part->npmax];
    
    
}

void particle_aux_deallocate(struct particles_aux* part_aux)
{
    // deallocate auxiliary particle variables needed for particle interpolation
    delete [] part_aux->rho_p;
    delete [] part_aux->Jx;
    delete [] part_aux->Jy;
    delete [] part_aux->Jz;
    delete [] part_aux->pxx;
    delete [] part_aux->pxy;
    delete [] part_aux->pxz;
    delete [] part_aux->pyy;
    delete [] part_aux->pyz;
    delete [] part_aux->pzz;
    
    
}
