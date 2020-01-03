#include "Particles.h"

/** move particle array to GPU */
void particle_move2gpu(struct particles* part, struct particles** part_gpu);

/** move particle array to CPU */
void particle_move2cpu(struct particles* part_gpu, struct particles* part);

/** deallocate */
void particle_deallocate_gpu(struct particles* part_gpu);


/** move EMfield to GPU */
void emfield_move2gpu(struct EMfield* field, struct EMfield** field_gpu, struct grid* grd);

/** move EMfield to CPU */
void emfield_move2cpu(struct EMfield* field_gpu, struct EMfield* field, struct grid* grd);

/** deallocate */
void emfield_deallocate_gpu(struct EMfield* field_gpu);


/** move grid to GPU */
void grid_move2gpu(struct grid* grd, struct grid** grd_gpu);

/** move grid to CPU */
void grid_move2cpu(struct grid* grd_gpu, struct grid* grd);

/** deallocate */
void grid_deallocate_gpu(struct grid* grd_gpu);

/**
* Allocate GPU memory for interpDensSpecies
* 
* @param ids_tmp interpDensSpecies struct on the host, but pointers will be linked to device addresses
* @param ids_gpu interpDensSpecies struct on the device
* @param grd grid structure
*/
void ids_allocate_gpu(struct interpDensSpecies* ids_tmp, struct interpDensSpecies** ids_gpu, struct grid* grd);

/**
* move interpDensSpecies to GPU
* 
* @param ids interpDensSpecies struct on the host
* @param ids_tmp interpDensSpecies struct on the host containing device pointers
* @param grd grid structure
*/
void ids_move2gpu(struct interpDensSpecies* ids, struct interpDensSpecies* ids_tmp, struct grid* grd);

/**
* move interpDensSpecies to CPU
* 
* @param ids_tmp interpDensSpecies struct on the host containing device pointers
* @param ids interpDensSpecies struct on the host
* @param grd grid structure
*/
void ids_move2cpu(struct interpDensSpecies* ids_tmp, struct interpDensSpecies* ids, struct grid* grd);

/** deallocate */
void ids_deallocate_gpu(struct interpDensSpecies* ids_gpu);