#ifndef EMFIELD_AUX_H
#define EMFIELD_AUX_H

#include "Alloc.h"
#include "Grid.h"

/** structure with auxiliary field quantities like potentials or quantities defined at centers  */
struct EMfield_aux {
    
    
    /* Electrostatic potential defined on central points*/
    FPfield*** Phi;
    FPfield* Phi_flat;

    /* Electric field at time theta */
    FPfield*** Exth;
    FPfield* Exth_flat;

    FPfield*** Eyth;
    FPfield* Eyth_flat;

    FPfield*** Ezth;
    FPfield* Ezth_flat;

    /* Magnetic field defined on nodes: last index is component - Centers */
    FPfield*** Bxc;
    FPfield* Bxc_flat;
    FPfield*** Byc;
    FPfield* Byc_flat;
    FPfield*** Bzc;
    FPfield* Bzc_flat;
    
};

/** allocate electric and magnetic field */
void field_aux_allocate(struct grid*, struct EMfield_aux*);

/** deallocate */
void field_aux_deallocate(struct grid*, struct EMfield_aux*);

#endif
