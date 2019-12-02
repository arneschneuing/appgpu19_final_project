#include "EMfield_aux.h"

/** allocate electric and magnetic field */
void field_aux_allocate(struct grid* grd, struct EMfield_aux* field_aux)
{
    // Electrostatic potential
    field_aux->Phi = newArr3<FPfield>(&field_aux->Phi_flat, grd->nxc, grd->nyc, grd->nzc);
    
    // allocate 3D arrays
    field_aux->Exth  = newArr3<FPfield>(&field_aux->Exth_flat, grd->nxn, grd->nyn, grd->nzn);
    field_aux->Eyth  = newArr3<FPfield>(&field_aux->Eyth_flat, grd->nxn, grd->nyn, grd->nzn);
    field_aux->Ezth  = newArr3<FPfield>(&field_aux->Ezth_flat, grd->nxn, grd->nyn, grd->nzn);

    // B on centers
    field_aux->Bxc = newArr3<FPfield>(&field_aux->Bxc_flat, grd->nxc, grd->nyc, grd->nzc);
    field_aux->Byc = newArr3<FPfield>(&field_aux->Byc_flat, grd->nxc, grd->nyc, grd->nzc);
    field_aux->Bzc = newArr3<FPfield>(&field_aux->Bzc_flat, grd->nxc, grd->nyc, grd->nzc);
}

/** deallocate */
void field_aux_deallocate(struct grid* grd, struct EMfield_aux* field_aux)
{
    // Eth
    delArr3(field_aux->Exth, grd->nxn, grd->nyn);
    delArr3(field_aux->Eyth, grd->nxn, grd->nyn);
    delArr3(field_aux->Ezth, grd->nxn, grd->nyn);

    // Bc
    delArr3(field_aux->Bxc, grd->nxc, grd->nyc);
    delArr3(field_aux->Byc, grd->nxc, grd->nyc);
    delArr3(field_aux->Bzc, grd->nxc, grd->nyc);

    // Phi
    delArr3(field_aux->Phi, grd->nxc, grd->nyc);
}
