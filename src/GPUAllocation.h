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