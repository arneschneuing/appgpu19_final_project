#ifndef RW_IO_H
#define RW_IO_H

#include "ConfigFile.h"
#include "input_array.h"

/** read the inputfile given via the command line */
void readInputFile(struct parameters*, int, char**);

/** Print Simulation Parameters */
void printParameters(struct parameters*);

/** Save Simulation Parameters */
void saveParameters(struct parameters*);

void VTK_Write_Vectors(int, struct grid*, struct EMfield*);

void VTK_Write_Scalars(int, struct grid*, struct interpDensSpecies*, struct interpDensNet*);

#endif
