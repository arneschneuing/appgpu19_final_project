#include "GPUAllocation.h"

/** move particle array to GPU */
void particle_move2gpu(struct particles* part, struct particles* part_gpu)
{   
    // Allocate memory on the GPU
    cudaMalloc(&part_gpu, sizeof(particles)); 

    // Allocate arrays on the device
    FPpart* x_gpu;
    cudaMalloc(&x_gpu, sizeof(FPpart)*part->npmax);

    FPpart* y_gpu;
    cudaMalloc(&y_gpu, sizeof(FPpart)*part->npmax);

    FPpart* z_gpu;
    cudaMalloc(&z_gpu, sizeof(FPpart)*part->npmax);

    FPpart* u_gpu;
    cudaMalloc(&u_gpu, sizeof(FPpart)*part->npmax);

    FPpart* v_gpu;
    cudaMalloc(&v_gpu, sizeof(FPpart)*part->npmax);

    FPpart* w_gpu;
    cudaMalloc(&w_gpu, sizeof(FPpart)*part->npmax);

    FPinterp* q_gpu;
    cudaMalloc(&q_gpu, sizeof(FPinterp)*part->npmax);

    // Copy array values to the device
    cudaMemcpy(x_gpu, part->x, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, part->y, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(z_gpu, part->z, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(u_gpu, part->u, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(v_gpu, part->v, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu, part->w, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(q_gpu, part->q, sizeof(FPinterp)*part->npmax, cudaMemcpyHostToDevice);



    // Create temporary copy of host pointers
    FPpart* x_host = part->x;
    FPpart* y_host = part->y;
    FPpart* z_host = part->z;
    FPpart* u_host = part->u;
    FPpart* v_host = part->v;
    FPpart* w_host = part->w;
    FPinterp* q_host = part->q;

    // Point to device pointer in host struct
    part->x = x_gpu;
    part->y = y_gpu;
    part->z = z_gpu;
    part->u = u_gpu;
    part->v = v_gpu;
    part->w = w_gpu;
    part->q = q_gpu;

    // Move data to the GPU (pointers still pointing to host addresses)
    cudaMemcpy(part_gpu, part, sizeof(particles), cudaMemcpyHostToDevice); 

    // Restore host pointer
    part->x = x_host;
    part->y = y_host;
    part->z = z_host;
    part->u = u_host;
    part->v = v_host;
    part->w = w_host;
    part->q = q_host; 
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

    // Allocate arrays on the device
    FPfield* Ex_flat_gpu;
    cudaMalloc(&Ex_flat_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    FPfield* Ey_flat_gpu;
    cudaMalloc(&Ey_flat_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    FPfield* Ez_flat_gpu;
    cudaMalloc(&Ez_flat_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    FPfield* Bxn_flat_gpu;
    cudaMalloc(&Bxn_flat_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    FPfield* Byn_flat_gpu;
    cudaMalloc(&Byn_flat_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    FPfield* Bzn_flat_gpu;
    cudaMalloc(&Bzn_flat_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    // Copy array values to the device
    cudaMemcpy(Ex_flat_gpu, field->Ex_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(Ey_flat_gpu, field->Ey_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(Ez_flat_gpu, field->Ez_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(Bxn_flat_gpu, field->Bxn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(Byn_flat_gpu, field->Byn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(Bzn_flat_gpu, field->Bzn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    // Create temporary copy of host pointers
    FPfield* Ex_flat_host = field->Ex_flat;
    FPfield* Ey_flat_host = field->Ey_flat;
    FPfield* Ez_flat_host = field->Ez_flat;
    FPfield* Bxn_flat_host = field->Bxn_flat;
    FPfield* Byn_flat_host = field->Byn_flat;
    FPfield* Bzn_flat_host = field->Bzn_flat;

    // Point to device pointer in host struct
    field->Ex_flat = Ex_flat_gpu;
    field->Ey_flat = Ey_flat_gpu;
    field->Ez_flat = Ez_flat_gpu;
    field->Bxn_flat = Bxn_flat_gpu;
    field->Byn_flat = Byn_flat_gpu;
    field->Bzn_flat = Bzn_flat_gpu;


    // Move data to the GPU (pointers still pointing to host addresses)
    cudaMemcpy(field_gpu, field, sizeof(EMfield), cudaMemcpyHostToDevice); 

    // Restore host pointer
    field->Ex_flat = Ex_flat_host;
    field->Ey_flat = Ey_flat_host;
    field->Ez_flat = Ez_flat_host;
    field->Bxn_flat = Bxn_flat_host;
    field->Byn_flat = Byn_flat_host;
    field->Bzn_flat = Bzn_flat_host;   
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