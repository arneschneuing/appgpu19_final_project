#include "InterpDensSpecies.h"

/** allocated interpolated densities per species */
void interp_dens_species_allocate(struct grid* grd, struct interpDensSpecies* ids, int is)
{
    // set species ID
    ids->species_ID = is;
    
    // allocate 3D arrays
    // rho: 1
    ids->rhon = newPinnedArr3<FPinterp>(&ids->rhon_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    ids->rhoc = newPinnedArr3<FPinterp>(&ids->rhoc_flat, grd->nxc, grd->nyc, grd->nzc); // center
    // Jx: 2
    ids->Jx   = newPinnedArr3<FPinterp>(&ids->Jx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jy: 3
    ids->Jy   = newPinnedArr3<FPinterp>(&ids->Jy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jz: 4
    ids->Jz   = newPinnedArr3<FPinterp>(&ids->Jz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxx: 5
    ids->pxx  = newPinnedArr3<FPinterp>(&ids->pxx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxy: 6
    ids->pxy  = newPinnedArr3<FPinterp>(&ids->pxy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxz: 7
    ids->pxz  = newPinnedArr3<FPinterp>(&ids->pxz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyy: 8
    ids->pyy  = newPinnedArr3<FPinterp>(&ids->pyy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyz: 9
    ids->pyz  = newPinnedArr3<FPinterp>(&ids->pyz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pzz: 10
    ids->pzz  = newPinnedArr3<FPinterp>(&ids->pzz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    
}

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate(struct grid* grd, struct interpDensSpecies* ids)
{
    
    // deallocate 3D arrays
    delPinnedArr3(ids->rhon, grd->nxn, grd->nyn);
    delPinnedArr3(ids->rhoc, grd->nxc, grd->nyc);
    // deallocate 3D arrays: J - current
    delPinnedArr3(ids->Jx, grd->nxn, grd->nyn);
    delPinnedArr3(ids->Jy, grd->nxn, grd->nyn);
    delPinnedArr3(ids->Jz, grd->nxn, grd->nyn);
    // deallocate 3D arrays: pressure
    delPinnedArr3(ids->pxx, grd->nxn, grd->nyn);
    delPinnedArr3(ids->pxy, grd->nxn, grd->nyn);
    delPinnedArr3(ids->pxz, grd->nxn, grd->nyn);
    delPinnedArr3(ids->pyy, grd->nxn, grd->nyn);
    delPinnedArr3(ids->pyz, grd->nxn, grd->nyn);
    delPinnedArr3(ids->pzz, grd->nxn, grd->nyn);
    
    
}

/** deallocate interpolated densities per species */
void interpN2Crho(struct interpDensSpecies* ids, struct grid* grd){
    for (register int i = 1; i < grd->nxc - 1; i++)
        for (register int j = 1; j < grd->nyc - 1; j++)
            for (register int k = 1; k < grd->nzc - 1; k++){
                ids->rhoc[i][j][k] = .125 * (ids->rhon[i][j][k] + ids->rhon[i + 1][j][k] + ids->rhon[i][j + 1][k] + ids->rhon[i][j][k + 1] +
                                       ids->rhon[i + 1][j + 1][k]+ ids->rhon[i + 1][j][k + 1] + ids->rhon[i][j + 1][k + 1] + ids->rhon[i + 1][j + 1][k + 1]);
            }
}
