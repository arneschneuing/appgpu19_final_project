#include "TransArraySpace3D.h"

/** method to convert a 1D field in a 3D field not considering guard cells*/
void solver2phys(FPfield*** vectPhys, FPfield* vectSolver, int nx, int ny, int nz){
    for (register int i=1; i < nx-1; i++)
      for (register int j=1; j < ny-1; j++)
        for (register int k=1; k < nz-1; k++)
           vectPhys[i][j][k] = *vectSolver++;
          
}
/** method to convert a 1D field in a 3D field not considering guard cells*/
void solver2phys(FPfield*** vectPhys1, FPfield*** vectPhys2, FPfield*** vectPhys3, FPfield* vectSolver, int nx, int ny, int nz){
    for (register int i=1; i < nx-1; i++)
      for (register int j=1; j < ny-1; j++)
        for (register int k=1; k < nz-1; k++){
          vectPhys1[i][j][k] = *vectSolver++;
          vectPhys2[i][j][k] = *vectSolver++;
          vectPhys3[i][j][k] = *vectSolver++;
        }
}


/** method to convert a 3D field in a 1D field not considering guard cells*/
void phys2solver(FPfield* vectSolver, FPfield*** vectPhys, int nx, int ny, int nz){
     for (register int i=1; i < nx-1; i++)
      for (register int j=1; j < ny-1; j++)
        for (register int k=1; k < nz-1; k++)
           *vectSolver++ = vectPhys[i][j][k];
           

}
/** method to convert a 3D field in a 1D field not considering guard cells*/
void phys2solver(FPfield* vectSolver, FPfield*** vectPhys1, FPfield*** vectPhys2, FPfield*** vectPhys3, int nx, int ny, int nz){
     for (register int i=1; i < nx-1; i++)
      for (register int j=1; j < ny-1; j++)
        for (register int k=1; k < nz-1; k++){
           *vectSolver++ =  vectPhys1[i][j][k];
           *vectSolver++ =  vectPhys2[i][j][k];
           *vectSolver++ =  vectPhys3[i][j][k];
        }

}
