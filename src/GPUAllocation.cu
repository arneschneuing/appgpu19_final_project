#include "GPUAllocation.h"

/** move particle array to GPU */
void particle_move2gpu(struct particles* part, struct particles* part_gpu)
{   
    // Allocate memory on the GPU
    cudaMalloc(&part_gpu, sizeof(particles)); 

    // Allocate arrays on the device
    FPfield* x_gpu;
    cudaMalloc(&x_gpu, sizeof(FPpart)*part->npmax);

    // Copy array values to the device
    cudaMemcpy(x_gpu, part->x, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);

    // Create temporary copy of host pointers
    FPfield* x_host = part->x;

    // Point to device pointer in host struct
    part->x = x_gpu;

    // Move data to the GPU (pointers still pointing to host addresses)
    cudaMemcpy(part_gpu, part, sizeof(particles), cudaMemcpyHostToDevice); 

    // Restore host pointer
    part->x = x_host;

    std::cout << "Hello" << std::endl;
    
    // move particle arrays
    cudaMalloc(&part_gpu->x, sizeof(FPpart)*part->npmax);
    std::cout << "Hello" << std::endl;
    cudaMemcpy(part_gpu->x, part->x, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    std::cout << "Hello" << std::endl;

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
    cudaFree(field_gpu->Ex_flat);
    cudaFree(field_gpu->Ey_flat);
    cudaFree(field_gpu->Ez_flat);
    cudaFree(field_gpu->Bxn_flat);
    cudaFree(field_gpu->Byn_flat);
    cudaFree(field_gpu->Bzn_flat);
    cudaFree(field_gpu);
}


/** move grid to GPU */
void grid_move2gpu(struct grid* grd, struct grid* grd_gpu)
{   
    // Allocate memory on the GPU
    cudaMalloc(&grd_gpu, sizeof(particles)); 

    // Allocate arrays on the device
    FPfield* XN_flat_gpu;
    cudaMalloc(&XN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    FPfield* YN_flat_gpu;
    cudaMalloc(&YN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    
    FPfield* ZN_flat_gpu;
    cudaMalloc(&ZN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    // Copy array values to the device
    cudaMemcpy(XN_flat_gpu, grd->XN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(YN_flat_gpu, grd->YN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_flat_gpu, grd->ZN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    // Create temporary copy of host pointers
    FPfield* XN_flat_host = grd->ZN;
    FPfield* YN_flat_host = grd->YN;
    FPfield* ZN_flat_host = grd->ZN;

    // Point to device pointer in host struct
    grd->XN_flat = XN_flat_gpu;
    grd->YN_flat = YN_flat_gpu;
    grd->ZN_flat = ZN_flat_gpu;


    // Move data to the GPU (pointers still pointing to host addresses)
    cudaMemcpy(grd_gpu, grd, sizeof(grid), cudaMemcpyHostToDevice); 

    // Restore host pointer
    grid->XN_flat = XN_flat_host;
    grid->YN_flat = YN_flat_host;
    grid->ZN_flat = ZN_flat_host;
}

/** move grid to CPU */
void grid_move2cpu(struct grid* grd_gpu, struct grid* grd)
{    
    // move arrays
    cudaMemcpy(grd->XN_flat, grd_gpu->XN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(grd->YN_flat, grd_gpu->YN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(grd->ZN_flat, grd_gpu->ZN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
}

/** deallocate */
void grid_deallocate_gpu(struct grid* grd_gpu)
{
    // deallocate variables
    cudaFree(grd_gpu->XN_flat);
    cudaFree(grd_gpu->YN_flat);
    cudaFree(grd_gpu->ZN_flat);

    cudaFree(grd_gpu);
}