#include "GPUAllocation.h"

/** move particle array to GPU */
void particle_move2gpu(struct particles* part, struct particles** part_gpu, cudaStream_t stream, int offset, int num_elem)
{   
    // Basic allocations and copies only in the first stream
    if (offset == 0)
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

        // Create temporary copy of host pointers
        FPpart* x_host = part->x;
        FPpart* y_host = part->y;
        FPpart* z_host = part->z;
        FPpart* u_host = part->u;
        FPpart* v_host = part->v;
        FPpart* w_host = part->w;
        FPinterp* q_host = part->q;

        // Point to device pointers in host struct
        part->x = x_gpu;
        part->y = y_gpu;
        part->z = z_gpu;
        part->u = u_gpu;
        part->v = v_gpu;
        part->w = w_gpu;
        part->q = q_gpu;

        // Move data to the GPU 
        // cudaMemcpy is implicitly blocking so that we can be sure parallel execution will not start before critical variables are copied
        cudaMemcpy(*part_gpu, part, sizeof(particles), cudaMemcpyHostToDevice); 

        // Restore host pointers
        part->x = x_host;
        part->y = y_host;
        part->z = z_host;
        part->u = u_host;
        part->v = v_host;
        part->w = w_host;
        part->q = q_host; 
    } 

    // Copy array values to the device
    cudaMemcpyAsync(x_gpu+offset, part->x+offset, sizeof(FPpart)*num_elem, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(y_gpu+offset, part->y+offset, sizeof(FPpart)*num_elem, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(z_gpu+offset, part->z+offset, sizeof(FPpart)*num_elem, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(u_gpu+offset, part->u+offset, sizeof(FPpart)*num_elem, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(v_gpu+offset, part->v+offset, sizeof(FPpart)*num_elem, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(w_gpu+offset, part->w+offset, sizeof(FPpart)*num_elem, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(q_gpu+offset, part->q+offset, sizeof(FPinterp)*num_elem, cudaMemcpyHostToDevice, stream);
}

/** move particle array to CPU */
void particle_move2cpu(struct particles* part_gpu, struct particles* part, cudaStream_t stream, int offset, int num_elem)
{   
    // Basic copies only in the first stream
    if (offset == 0)
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
        // cudaMemcpy is implicitly blocking so that we can be sure parallel execution will not start before critical variables are copied
        cudaMemcpy(part, part_gpu, sizeof(particles), cudaMemcpyDeviceToHost);

        // Create temporary copy of device pointers
        FPpart* x_device = part->x;
        FPpart* y_device = part->y;
        FPpart* z_device = part->z;
        FPpart* u_device = part->u;
        FPpart* v_device = part->v;
        FPpart* w_device = part->w;
        FPinterp* q_device = part->q;

        // Restore host pointers
        part->x = x_host;
        part->y = y_host;
        part->z = z_host;
        part->u = u_host;
        part->v = v_host;
        part->w = w_host;
        part->q = q_host;
    }
    
    // move particle arrays
    cudaMemcpyAsync(part->x+offset, x_device+offset, sizeof(FPpart)*num_elem, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(part->y+offset, y_device+offset, sizeof(FPpart)*num_elem, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(part->z+offset, z_device+offset, sizeof(FPpart)*num_elem, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(part->u+offset, u_device+offset, sizeof(FPpart)*num_elem, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(part->v+offset, v_device+offset, sizeof(FPpart)*num_elem, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(part->w+offset, w_device+offset, sizeof(FPpart)*num_elem, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(part->q+offset, q_device+offset, sizeof(FPinterp)*num_elem, cudaMemcpyDeviceToHost, stream);
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

    // Point to device pointers in host struct
    field->Ex_flat = Ex_flat_gpu;
    field->Ey_flat = Ey_flat_gpu;
    field->Ez_flat = Ez_flat_gpu;
    field->Bxn_flat = Bxn_flat_gpu;
    field->Byn_flat = Byn_flat_gpu;
    field->Bzn_flat = Bzn_flat_gpu;


    // Move data to the GPU 
    cudaMemcpy(*field_gpu, field, sizeof(EMfield), cudaMemcpyHostToDevice); 

    // Restore host pointers
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

    // Restore host pointers
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

    // Point to device pointers in host struct
    grd->XN_flat = XN_flat_gpu;
    grd->YN_flat = YN_flat_gpu;
    grd->ZN_flat = ZN_flat_gpu;


    // Move data to the GPU 
    cudaMemcpy(*grd_gpu, grd, sizeof(grid), cudaMemcpyHostToDevice); 

    // Restore host pointers
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

    // Restore host pointers
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


/** move interpDensSpecies to GPU */
void ids_move2gpu(struct interpDensSpecies* ids, struct interpDensSpecies** ids_gpu, struct grid* grd)
{   
    // Allocate memory on the GPU
    cudaMalloc(ids_gpu, sizeof(interpDensSpecies)); 

    // Allocate arrays on the device
    FPinterp* rhon_flat_gpu;
    cudaMalloc(&rhon_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* rhoc_flat_gpu;
    cudaMalloc(&rhoc_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* Jx_flat_gpu;
    cudaMalloc(&Jx_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* Jy_flat_gpu;
    cudaMalloc(&Jy_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* Jz_flat_gpu;
    cudaMalloc(&Jz_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* pxx_flat_gpu;
    cudaMalloc(&pxx_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* pxy_flat_gpu;
    cudaMalloc(&pxy_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* pxz_flat_gpu;
    cudaMalloc(&pxz_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* pyy_flat_gpu;
    cudaMalloc(&pyy_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* pyz_flat_gpu;
    cudaMalloc(&pyz_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);

    FPinterp* pzz_flat_gpu;
    cudaMalloc(&pzz_flat_gpu, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn);


    // Copy array values to the device
    cudaMemcpy(rhon_flat_gpu, ids->rhon_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);    
    cudaMemcpy(rhoc_flat_gpu, ids->rhoc_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(Jx_flat_gpu, ids->Jx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(Jy_flat_gpu, ids->Jy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(Jz_flat_gpu, ids->Jz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(pxx_flat_gpu, ids->pxx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(pxy_flat_gpu, ids->pxy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(pxz_flat_gpu, ids->pxz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(pyy_flat_gpu, ids->pyy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(pyz_flat_gpu, ids->pyz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(pzz_flat_gpu, ids->pzz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 

    // Create temporary copy of host pointers
    FPinterp* rhon_flat_host = ids->rhon_flat;
    FPinterp* rhoc_flat_host = ids->rhoc_flat;
    FPinterp* Jx_flat_host = ids->Jx_flat;
    FPinterp* Jy_flat_host = ids->Jy_flat;
    FPinterp* Jz_flat_host = ids->Jz_flat;
    FPinterp* pxx_flat_host = ids->pxx_flat;
    FPinterp* pxy_flat_host = ids->pxy_flat;
    FPinterp* pxz_flat_host = ids->pxz_flat;
    FPinterp* pyy_flat_host = ids->pyy_flat;
    FPinterp* pyz_flat_host = ids->pyz_flat;
    FPinterp* pzz_flat_host = ids->pzz_flat;

    // Point to device pointers in host struct
    ids->rhon_flat = rhon_flat_gpu;
    ids->rhoc_flat = rhoc_flat_gpu;
    ids->Jx_flat = Jx_flat_gpu;
    ids->Jy_flat = Jy_flat_gpu;
    ids->Jz_flat = Jz_flat_gpu;
    ids->pxx_flat = pxx_flat_gpu;
    ids->pxy_flat = pxy_flat_gpu;
    ids->pxz_flat = pxz_flat_gpu;
    ids->pyy_flat = pyy_flat_gpu;
    ids->pyz_flat = pyz_flat_gpu;
    ids->pzz_flat = pzz_flat_gpu;

    // Move data to the GPU 
    cudaMemcpy(*ids_gpu, ids, sizeof(interpDensSpecies), cudaMemcpyHostToDevice); 

    // Restore host pointers
    ids->rhon_flat = rhon_flat_host;
    ids->rhoc_flat = rhoc_flat_host;
    ids->Jx_flat = Jx_flat_host;
    ids->Jy_flat = Jy_flat_host;
    ids->Jz_flat = Jz_flat_host;
    ids->pxx_flat = pxx_flat_host;
    ids->pxy_flat = pxy_flat_host;
    ids->pxz_flat = pxz_flat_host;
    ids->pyy_flat = pyy_flat_host;
    ids->pyz_flat = pyz_flat_host;
    ids->pzz_flat = pzz_flat_host;
}

/** move interpDensSpecies to CPU */
void ids_move2cpu(struct interpDensSpecies* ids_gpu, struct interpDensSpecies* ids, struct grid* grd)
{    
    // Create temporary copy of host pointers
    FPinterp* rhon_flat_host = ids->rhon_flat;
    FPinterp* rhoc_flat_host = ids->rhoc_flat;
    FPinterp* Jx_flat_host = ids->Jx_flat;
    FPinterp* Jy_flat_host = ids->Jy_flat;
    FPinterp* Jz_flat_host = ids->Jz_flat;
    FPinterp* pxx_flat_host = ids->pxx_flat;
    FPinterp* pxy_flat_host = ids->pxy_flat;
    FPinterp* pxz_flat_host = ids->pxz_flat;
    FPinterp* pyy_flat_host = ids->pyy_flat;
    FPinterp* pyz_flat_host = ids->pyz_flat;
    FPinterp* pzz_flat_host = ids->pzz_flat;

    // Move data to the CPU
    cudaMemcpy(ids, ids_gpu, sizeof(interpDensSpecies), cudaMemcpyDeviceToHost);

    // Create temporary copy of device pointers
    FPinterp* rhon_flat_device = ids->rhon_flat;
    FPinterp* rhoc_flat_device = ids->rhoc_flat;
    FPinterp* Jx_flat_device = ids->Jx_flat;
    FPinterp* Jy_flat_device = ids->Jy_flat;
    FPinterp* Jz_flat_device = ids->Jz_flat;
    FPinterp* pxx_flat_device = ids->pxx_flat;
    FPinterp* pxy_flat_device = ids->pxy_flat;
    FPinterp* pxz_flat_device = ids->pxz_flat;
    FPinterp* pyy_flat_device = ids->pyy_flat;
    FPinterp* pyz_flat_device = ids->pyz_flat;
    FPinterp* pzz_flat_device = ids->pzz_flat;

    // Restore host pointers
    ids->rhon_flat = rhon_flat_host;
    ids->rhoc_flat = rhoc_flat_host;
    ids->Jx_flat = Jx_flat_host;
    ids->Jy_flat = Jy_flat_host;
    ids->Jz_flat = Jz_flat_host;
    ids->pxx_flat = pxx_flat_host;
    ids->pxy_flat = pxy_flat_host;
    ids->pxz_flat = pxz_flat_host;
    ids->pyy_flat = pyy_flat_host;
    ids->pyz_flat = pyz_flat_host;
    ids->pzz_flat = pzz_flat_host;

    // move arrays
    cudaMemcpy(ids->rhon_flat, rhon_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->rhoc_flat, rhoc_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jx_flat, Jx_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jy_flat, Jy_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jz_flat, Jz_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxx_flat, pxx_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxy_flat, pxy_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxz_flat, pxz_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyy_flat, pyy_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyz_flat, pyz_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pzz_flat, pzz_flat_device, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
}

/** deallocate */
void ids_deallocate_gpu(struct interpDensSpecies* ids_gpu)
{
    // Create temporary struct
    interpDensSpecies* ids_tmp = new interpDensSpecies;
    cudaMemcpy(ids_tmp, ids_gpu, sizeof(interpDensSpecies), cudaMemcpyDeviceToHost);

    // deallocate variables
    cudaFree(ids_tmp->rhon_flat);
    cudaFree(ids_tmp->rhoc_flat);
    cudaFree(ids_tmp->Jx_flat);
    cudaFree(ids_tmp->Jy_flat);
    cudaFree(ids_tmp->Jz_flat);
    cudaFree(ids_tmp->pxx_flat);
    cudaFree(ids_tmp->pxy_flat);
    cudaFree(ids_tmp->pxz_flat);
    cudaFree(ids_tmp->pyy_flat);
    cudaFree(ids_tmp->pyz_flat);
    cudaFree(ids_tmp->pzz_flat);

    cudaFree(ids_gpu);
}