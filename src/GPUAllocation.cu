#include "Particles.h"

/** move particle array to GPU */
void particle_move2gpu(struct particles* part, struct particles* part_gpu)
{   
    // Allocate memory on the GPU
    cudaMalloc(&part_gpu, sizeof(particles));

    // Move data to the GPU
    cudaMemcpy(part_gpu, part, sizeof(particles), cudaMemcpyHostToDevice);    
    
    // move particle arrays
    cudaMalloc(&part_gpu->x, sizeof(FPpart)*part->npmax);
    cudaMemcpy(part_gpu->x, part->x, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);

    cudaMalloc(&part_gpu->y, sizeof(FPpart)*part->npmax);
    cudaMemcpy(part_gpu->y, part->y, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);

    cudaMalloc(&part_gpu->z, sizeof(FPpart)*part->npmax);
    cudaMemcpy(part_gpu->z, part->z, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);

    cudaMalloc(&part_gpu->u, sizeof(FPpart)*part->npmax);
    cudaMemcpy(part_gpu->u, part->u, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);

    cudaMalloc(&part_gpu->v, sizeof(FPpart)*part->npmax);
    cudaMemcpy(part_gpu->v, part->v, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);

    cudaMalloc(&part_gpu->w, sizeof(FPpart)*part->npmax);
    cudaMemcpy(part_gpu->w, part->w, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);

    cudaMalloc(&part_gpu->q, sizeof(FPinterp)*part->npmax);
    cudaMemcpy(part_gpu->q, part->q, sizeof(FPinterp)*part->npmax, cudaMemcpyHostToDevice);    
}

/** move particle array to CPU */
void particle_move2cpu(struct particles* part_gpu, struct particles* part)
{   
    // Move data to the CPU
    cudaMemcpy(part, part_gpu, sizeof(particles), cudaMemcpyDeviceToHost);    
    
    // move particle arrays
    cudaMemcpy(part->x, part_gpu->x, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->y, part_gpu->y, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->z, part_gpu->z, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->u, part_gpu->u, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->v, part_gpu->v, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->w, part_gpu->w, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->q, part_gpu->q, sizeof(FPinterp)*part->npmax, cudaMemcpyDeviceToHost);    
}

/** deallocate */
void particle_deallocate_gpu(struct particles* part_gpu)
{
    // deallocate particle variables
    cudaFree(part_gpu->x);
    cudaFree(part_gpu->y);
    cudaFree(part_gpu->z);
    cudaFree(part_gpu->u);
    cudaFree(part_gpu->v);
    cudaFree(part_gpu->w);
    cudaFree(part_gpu->q);
    cudaFree(part_gpu);
}


/** move EMfield to GPU */
void emfield_move2gpu(struct EMfield* field, struct EMfield* field_gpu, struct grid* grd)
{   
    // Allocate memory on the GPU
    cudaMalloc(&field_gpu, sizeof(EMfield));   
    
    // Copy arrays
    cudaMalloc(&field_gpu->Ex_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMemcpy(field_gpu->Ex_flat, field->Ex_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    cudaMalloc(&field_gpu->Ey_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMemcpy(field_gpu->Ey_flat, field->Ey_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    cudaMalloc(&field_gpu->Ez_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMemcpy(field_gpu->Ez_flat, field->Ez_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    cudaMalloc(&field_gpu->Bxn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMemcpy(field_gpu->Bxn_flat, field->Bxn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    cudaMalloc(&field_gpu->Byn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMemcpy(field_gpu->Byn_flat, field->Byn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    cudaMalloc(&field_gpu->Bzn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMemcpy(field_gpu->Bzn_flat, field->Bzn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
   
}

/** move EMfield to CPU */
void emfield_move2cpu(struct EMfield* field_gpu, struct EMfield* field, struct grid* grd)
{    
    // move arrays
    cudaMemcpy(field->Ex_flat, field_gpu->Ex_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Ey_flat, field_gpu->Ey_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Ez_flat, field_gpu->Ez_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Bxn_flat, field_gpu->Bxn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Byn_flat, field_gpu->Byn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Bzn_flat, field_gpu->Bzn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);  
}

/** deallocate */
void emfield_deallocate_gpu(struct EMfield* field_gpu)
{
    // deallocate variables
    cudaFree(field_gpu->Ex);
    cudaFree(field_gpu->Ey);
    cudaFree(field_gpu->Ez);
    cudaFree(field_gpu->Bxn);
    cudaFree(field_gpu->Byn);
    cudaFree(field_gpu->Bzn);
    cudaFree(field_gpu);
}