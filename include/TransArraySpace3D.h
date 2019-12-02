#ifndef TRANSARRAYSPACE3D_H
#define TRANSARRAYSPACE3D_H

#include "PrecisionTypes.h"

/** method to convert a 1D field in a 3D field not considering guard cells*/
void solver2phys(FPfield*** vectPhys, FPfield* vectSolver, int nx, int ny, int nz);
/** method to convert a 1D field in a 3D field not considering guard cells*/
void solver2phys(FPfield*** vectPhys1, FPfield*** vectPhys2, FPfield*** vectPhys3, FPfield* vectSolver, int nx, int ny, int nz);


/** method to convert a 3D field in a 1D field not considering guard cells*/
void phys2solver(FPfield* vectSolver, FPfield*** vectPhys, int nx, int ny, int nz);
/** method to convert a 3D field in a 1D field not considering guard cells*/
void phys2solver(FPfield* vectSolver, FPfield*** vectPhys1, FPfield*** vectPhys2, FPfield*** vectPhys3, int nx, int ny, int nz);

#endif
