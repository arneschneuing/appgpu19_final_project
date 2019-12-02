#ifndef GRID_H
#define GRID_H

#include <iostream>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"

/** Grid Data */
struct grid {
    /** number of cells - X direction, including + 2 (guard cells) */
    int nxc;
    /** number of nodes - X direction, including + 2 extra nodes for guard cells */
    int nxn;
    /** number of cell - Y direction, including + 2 (guard cells) */
    int nyc;
    /** number of nodes - Y direction, including + 2 extra nodes for guard cells */
    int nyn;
    /** number of cell - Z direction, including + 2 (guard cells) */
    int nzc;
    /** number of nodes - Z direction, including + 2 extra nodes for guard cells */
    int nzn;
    /** dx = space step - X direction */
    double dx;
    /** dy = space step - Y direction */
    double dy;
    /** dz = space step - Z direction */
    double dz;
    /** invdx = 1/dx */
    FPfield invdx;
    /** invdy = 1/dy */
    FPfield invdy;
    /** invdz = 1/dz */
    FPfield invdz;
    /** invol = inverse of volume*/
    FPfield invVOL;
    /** local grid boundaries coordinate  */
    double xStart, xEnd, yStart, yEnd, zStart, zEnd;
    /** domain size */
    double Lx, Ly, Lz;
    
    /** Periodicity for fields X **/
    bool PERIODICX;
    /** Periodicity for fields Y **/
    bool PERIODICY;
    /** Periodicity for fields Z **/
    bool PERIODICZ;
    
    // Nodes coordinate
    /** coordinate node X */
    FPfield* XN_flat;
    FPfield*** XN;
    /** coordinate node Y */
    FPfield* YN_flat;
    FPfield*** YN;
    /** coordinate node Z */
    FPfield* ZN_flat;
    FPfield*** ZN;
    
    
};

/** Set up the grid quantities */
void setGrid(struct parameters*, struct grid*);

/** Set up the grid quantities */
void printGrid(struct grid*);

/** allocate electric and magnetic field */
void grid_deallocate(struct grid*);

/** interpolation Node to Center */
void interpN2Cfield(FPfield***, FPfield***, FPfield***, FPfield***, FPfield***, FPfield***, struct grid*);
    
/** interpolation Node to Center */
void interpC2Ninterp(FPinterp***, FPinterp***, struct grid*);

/** interpolation Node to Center */
void interpC2Nfield(FPfield***, FPfield***, struct grid*);

/** interpolation Node to Center */
void interpN2Cinterp(FPinterp***, FPinterp***, struct grid*);

/** calculate gradient on nodes, given a scalar field defined on central points  */
void gradC2N(FPfield***, FPfield***, FPfield***, FPfield***, grid*);

/** calculate gradient on nodes, given a scalar field defined on central points  */
void gradN2C(FPfield***, FPfield***, FPfield***, FPfield***, grid*);

/** calculate divergence on central points, given a vector field defined on nodes  */
void divN2C(FPfield***, FPfield***, FPfield***, FPfield***, grid*);

/** calculate divergence on central points, given a Tensor field defined on nodes  */
void divSymmTensorN2C(FPinterp***, FPinterp***, FPinterp***, FPinterp***, FPinterp***, FPinterp***, FPinterp***, FPinterp***, FPinterp***, grid*);

/** calculate divergence on nodes, given a vector field defined on central points  */
void divC2N(FPfield***, FPfield***, FPfield***, FPfield***, grid*);

/** calculate curl on nodes, given a vector field defined on central points  */
void curlC2N(FPfield***, FPfield***, FPfield***, FPfield***, FPfield***, FPfield***, grid*);

/** calculate curl on central points, given a vector field defined on nodes  */
void curlN2C(FPfield***, FPfield***, FPfield***, FPfield***, FPfield***, FPfield***, grid*);

/** calculate laplacian on nodes, given a scalar field defined on nodes */
void lapN2N(FPfield***, FPfield***, grid*);

/** calculate laplacian on central points, given a scalar field defined on central points */
void lapC2C(FPfield***, FPfield***, grid*);


#endif
