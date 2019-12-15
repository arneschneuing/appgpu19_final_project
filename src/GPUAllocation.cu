#include "GPUAllocation.h"

/** move particle array to GPU */
void particle_move2gpu(struct particles* part, struct particles** part_gpu)
{   
    // Allocate memory on the GPU
    cudaMalloc(part_gpu, sizeof(particles)); 

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
    cudaMemcpy(*part_gpu, part, sizeof(particles), cudaMemcpyHostToDevice); 

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
    // Create temporary copy of host pointers
    FPpart* x_host = part->x;
    FPpart* y_host = part->y;
    FPpart* z_host = part->z;
    FPpart* u_host = part->u;
    FPpart* v_host = part->v;
    FPpart* w_host = part->w;
    FPinterp* q_host = part->q;

    // Move data to the CPU
    cudaMemcpy(part, part_gpu, sizeof(particles), cudaMemcpyDeviceToHost);

    // Create temporary copy of device pointers
    FPpart* x_device = part->x;
    FPpart* y_device = part->y;
    FPpart* z_device = part->z;
    FPpart* u_device = part->u;
    FPpart* v_device = part->v;
    FPpart* w_device = part->w;
    FPinterp* q_device = part->q;

    // Restore host pointer
    part->x = x_host;
    part->y = y_host;
    part->z = z_host;
    part->u = u_host;
    part->v = v_host;
    part->w = w_host;
    part->q = q_host;
    
    // move particle arrays
    cudaMemcpy(part->x, x_device, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->y, y_device, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->z, z_device, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->u, u_device, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->v, v_device, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->w, w_device, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);

    cudaMemcpy(part->q, q_device, sizeof(FPinterp)*part->npmax, cudaMemcpyDeviceToHost);
}

/** deallocate */
void particle_deallocate_gpu(struct particles* part_gpu)
{
    // Create temporary struct
    particles* part_tmp = new particles;
    cudaMemcpy(part_tmp, part_gpu, sizeof(particles), cudaMemcpyDeviceToHost);

    // deallocate particle variables
    cudaFree(part_tmp->x);
    cudaFree(part_tmp->y);
    cudaFree(part_tmp->z);
    cudaFree(part_tmp->u);
    cudaFree(part_tmp->v);
    cudaFree(part_tmp->w);
    cudaFree(part_tmp->q);
    cudaFree(part_gpu);
}


/** move EMfield to GPU */
void emfield_move2gpu(struct EMfield* field, struct EMfield** field_gpu, struct grid* grd)
{   
    // Allocate memory on the GPU
    cudaMalloc(field_gpu, sizeof(EMfield));

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
    cudaMemcpy(*field_gpu, field, sizeof(EMfield), cudaMemcpyHostToDevice); 

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
    // Create temporary copy of host pointers
    FPfield* Ex_flat_host = field->Ex_flat;
    FPfield* Ey_flat_host = field->Ey_flat;
    FPfield* Ez_flat_host = field->Ez_flat;
    FPfield* Bxn_flat_host = field->Bxn_flat;
    FPfield* Byn_flat_host = field->Byn_flat;
    FPfield* Bzn_flat_host = field->Bzn_flat;

    // Move data to the CPU
    cudaMemcpy(field, field_gpu, sizeof(EMfield), cudaMemcpyDeviceToHost);

    // Create temporary copy of device pointers
    FPfield* Ex_flat_device = field->Ex_flat;
    FPfield* Ey_flat_device = field->Ey_flat;
    FPfield* Ez_flat_device = field->Ez_flat;
    FPfield* Bxn_flat_device = field->Bxn_flat;
    FPfield* Byn_flat_device = field->Byn_flat;
    FPfield* Bzn_flat_device = field->Bzn_flat;

    // Restore host pointer
    field->Ex_flat = Ex_flat_host;
    field->Ey_flat = Ey_flat_host;
    field->Ez_flat = Ez_flat_host;
    field->Bxn_flat = Bxn_flat_host;
    field->Byn_flat = Byn_flat_host;
    field->Bzn_flat = Bzn_flat_host;

    // move arrays
    cudaMemcpy(field->Ex_flat, Ex_flat_device, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Ey_flat, Ey_flat_device, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Ez_flat, Ez_flat_device, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Bxn_flat, Bxn_flat_device, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Byn_flat, Byn_flat_device, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Bzn_flat, Bzn_flat_device, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);  
}

/** deallocate */
void emfield_deallocate_gpu(struct EMfield* field_gpu)
{
    // Create temporary struct
    EMfield* field_tmp = new EMfield;
    cudaMemcpy(field_tmp, field_gpu, sizeof(EMfield), cudaMemcpyDeviceToHost);

    // deallocate variables
    cudaFree(field_tmp->Ex_flat);
    cudaFree(field_tmp->Ey_flat);
    cudaFree(field_tmp->Ez_flat);
    cudaFree(field_tmp->Bxn_flat);
    cudaFree(field_tmp->Byn_flat);
    cudaFree(field_tmp->Bzn_flat);
    cudaFree(field_gpu);
}


/** move grid to GPU */
void grid_move2gpu(struct grid* grd, struct grid** grd_gpu)
{   
    // Allocate memory on the GPU
    cudaMalloc(grd_gpu, sizeof(grid)); 

    // Allocate arrays on the device
    FPfield* XN_flat_gpu;
    cudaMalloc(&XN_flat_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    FPfield* YN_flat_gpu;
    cudaMalloc(&YN_flat_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    
    FPfield* ZN_flat_gpu;
    cudaMalloc(&ZN_flat_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    // Copy array values to the device
    cudaMemcpy(XN_flat_gpu, grd->XN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(YN_flat_gpu, grd->YN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_flat_gpu, grd->ZN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    // Create temporary copy of host pointers
    FPfield* XN_flat_host = grd->XN_flat;
    FPfield* YN_flat_host = grd->YN_flat;
    FPfield* ZN_flat_host = grd->ZN_flat;

    // Point to device pointer in host struct
    grd->XN_flat = XN_flat_gpu;
    grd->YN_flat = YN_flat_gpu;
    grd->ZN_flat = ZN_flat_gpu;


    // Move data to the GPU (pointers still pointing to host addresses)
    cudaMemcpy(*grd_gpu, grd, sizeof(grid), cudaMemcpyHostToDevice); 

    // Restore host pointer
    grd->XN_flat = XN_flat_host;
    grd->YN_flat = YN_flat_host;
    grd->ZN_flat = ZN_flat_host;
}

/** move grid to CPU */
void grid_move2cpu(struct grid* grd_gpu, struct grid* grd)
{    
    // Create temporary copy of host pointers
    FPfield* XN_flat_host = grd->XN_flat;
    FPfield* YN_flat_host = grd->YN_flat;
    FPfield* ZN_flat_host = grd->ZN_flat;

    // Move data to the CPU
    cudaMemcpy(grd, grd_gpu, sizeof(grid), cudaMemcpyDeviceToHost);

    // Create temporary copy of device pointers
    FPfield* XN_flat_device = grd->XN_flat;
    FPfield* YN_flat_device = grd->YN_flat;
    FPfield* ZN_flat_device = grd->ZN_flat;

    // Restore host pointer
    grd->XN_flat = XN_flat_host;
    grd->YN_flat = YN_flat_host;
    grd->ZN_flat = ZN_flat_host;

    // move arrays
    cudaMemcpy(grd->XN_flat, XN_flat_device, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(grd->YN_flat, YN_flat_device, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

    cudaMemcpy(grd->ZN_flat, ZN_flat_device, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
}

/** deallocate */
void grid_deallocate_gpu(struct grid* grd_gpu)
{
    // Create temporary struct
    grid* grd_tmp = new grid;
    cudaMemcpy(grd_tmp, grd_gpu, sizeof(grid), cudaMemcpyDeviceToHost);

    // deallocate variables
    cudaFree(grd_tmp->XN_flat);
    cudaFree(grd_tmp->YN_flat);
    cudaFree(grd_tmp->ZN_flat);

    cudaFree(grd_gpu);
}