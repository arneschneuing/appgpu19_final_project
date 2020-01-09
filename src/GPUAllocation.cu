#include "GPUAllocation.h"

/**
 * Checks if there has been a memory allocation error. 
 * The method will exit the program when an error is found.
 */
 inline void checkMemAlloc()
 {
     cudaError_t cudaError = cudaGetLastError();
     
     if(cudaError == cudaErrorMemoryAllocation)
     {
         printf("The API call failed because it was unable to allocate enough memory to perform the requested operation.\n
                Try to increase the number of batches (nob) in the input file!");
         exit(-1);
     }
 }

/**
* Allocate GPU memory for particle struct
* 
* @param part_tmp particle struct on the host, but pointers will be linked to device addresses
* @param part_gpu particle struct on the device
* @param npmax maximum number of particles
*/
void particle_allocate_gpu(struct particles* part_tmp, struct particles** part_gpu, int npmax)
{
    // Allocate memory on the GPU
    cudaMalloc(part_gpu, sizeof(particles)); 

    // Allocate arrays on the device
    FPpart* x_gpu;
    cudaMalloc(&x_gpu, sizeof(FPpart)*npmax);

    FPpart* y_gpu;
    cudaMalloc(&y_gpu, sizeof(FPpart)*npmax);

    FPpart* z_gpu;
    cudaMalloc(&z_gpu, sizeof(FPpart)*npmax);

    FPpart* u_gpu;
    cudaMalloc(&u_gpu, sizeof(FPpart)*npmax);

    FPpart* v_gpu;
    cudaMalloc(&v_gpu, sizeof(FPpart)*npmax);

    FPpart* w_gpu;
    cudaMalloc(&w_gpu, sizeof(FPpart)*npmax);

    FPinterp* q_gpu;
    cudaMalloc(&q_gpu, sizeof(FPinterp)*npmax);

    // Check if memory allocation was successful
    checkMemAlloc();

    // Point to device pointers in host struct
    part_tmp->x = x_gpu;
    part_tmp->y = y_gpu;
    part_tmp->z = z_gpu;
    part_tmp->u = u_gpu;
    part_tmp->v = v_gpu;
    part_tmp->w = w_gpu;
    part_tmp->q = q_gpu;

    // Move data to the GPU (pointers are pointing to device addresses)
    cudaMemcpy(*part_gpu, part_tmp, sizeof(particles), cudaMemcpyHostToDevice); 
}

/** 
* move particle arrays to GPU 
* 
* @param part particle struct on the host
* @param part_tmp particle struct on the host containing device pointers
* @param part_gpu particle struct on the device
*/
void particle_move2gpu(struct particles* part, struct particles* part_tmp, struct particles** part_gpu)
{   
    // update relevant scalar values in temporary structure
    part_tmp->nop = part->nop;
    part_tmp->qom = part->qom;
    part_tmp->n_sub_cycles = part->n_sub_cycles;
    part_tmp->NiterMover = part->NiterMover;

    // Copy array values to the device
    cudaMemcpy(part_tmp->x, part->x, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(part_tmp->y, part->y, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(part_tmp->z, part->z, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(part_tmp->u, part->u, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(part_tmp->v, part->v, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(part_tmp->w, part->w, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(part_tmp->q, part->q, sizeof(FPinterp)*part->npmax, cudaMemcpyHostToDevice);

    // Move scalar values (and pointer addresses) to the GPU
    cudaMemcpy(*part_gpu, part_tmp, sizeof(particles), cudaMemcpyHostToDevice); 
}

/** 
* move particle array to CPU 
* @param part_tmp particles struct on the host containing device pointers
* @param part particles struct on the host
*/
void particle_move2cpu(struct particles* part_tmp, struct particles* part)
{     
    // move particle arrays
    cudaMemcpy(part->x, part_tmp->x, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, part_tmp->y, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, part_tmp->z, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, part_tmp->u, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, part_tmp->v, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, part_tmp->w, sizeof(FPpart)*part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->q, part_tmp->q, sizeof(FPinterp)*part->npmax, cudaMemcpyDeviceToHost);
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
    delete[] part_tmp;
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


    // Move data to the GPU (pointers are pointing to device addresses)
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
    delete[] field_tmp; 
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


    // Move data to the GPU (pointers are pointing to device addresses)
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
    delete[] grd_tmp;
}

/**
* Allocate GPU memory for interpDensSpecies
* 
* @param ids_tmp interpDensSpecies struct on the host, but pointers will be linked to device addresses
* @param ids_gpu interpDensSpecies struct on the device
* @param grd grid structure
*/
void ids_allocate_gpu(struct interpDensSpecies* ids_tmp, struct interpDensSpecies** ids_gpu, struct grid* grd)
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

    // Point to device pointers in host struct
    ids_tmp->rhon_flat = rhon_flat_gpu;
    ids_tmp->rhoc_flat = rhoc_flat_gpu;
    ids_tmp->Jx_flat = Jx_flat_gpu;
    ids_tmp->Jy_flat = Jy_flat_gpu;
    ids_tmp->Jz_flat = Jz_flat_gpu;
    ids_tmp->pxx_flat = pxx_flat_gpu;
    ids_tmp->pxy_flat = pxy_flat_gpu;
    ids_tmp->pxz_flat = pxz_flat_gpu;
    ids_tmp->pyy_flat = pyy_flat_gpu;
    ids_tmp->pyz_flat = pyz_flat_gpu;
    ids_tmp->pzz_flat = pzz_flat_gpu;

    // Move correct pointer addresses to the GPU
    cudaMemcpy(*ids_gpu, ids_tmp, sizeof(interpDensSpecies), cudaMemcpyHostToDevice); 
}

/**
* move interpDensSpecies to GPU
* 
* @param ids interpDensSpecies struct on the host
* @param ids_tmp interpDensSpecies struct on the host containing device pointers
* @param grd grid structure
*/
void ids_move2gpu(struct interpDensSpecies* ids, struct interpDensSpecies* ids_tmp, struct grid* grd)
{   
    // Copy array values to the device
    cudaMemcpy(ids_tmp->rhon_flat, ids->rhon_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);    
    cudaMemcpy(ids_tmp->rhoc_flat, ids->rhoc_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(ids_tmp->Jx_flat, ids->Jx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(ids_tmp->Jy_flat, ids->Jy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(ids_tmp->Jz_flat, ids->Jz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(ids_tmp->pxx_flat, ids->pxx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(ids_tmp->pxy_flat, ids->pxy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(ids_tmp->pxz_flat, ids->pxz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(ids_tmp->pyy_flat, ids->pyy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(ids_tmp->pyz_flat, ids->pyz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
    cudaMemcpy(ids_tmp->pzz_flat, ids->pzz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice); 
}

/**
* move interpDensSpecies to CPU
* 
* @param ids_tmp interpDensSpecies struct on the host containing device pointers
* @param ids interpDensSpecies struct on the host
* @param grd grid structure
*/
void ids_move2cpu(struct interpDensSpecies* ids_tmp, struct interpDensSpecies* ids, struct grid* grd)
{    
    // move arrays
    cudaMemcpy(ids->rhon_flat, ids_tmp->rhon_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->rhoc_flat, ids_tmp->rhoc_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jx_flat, ids_tmp->Jx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jy_flat, ids_tmp->Jy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jz_flat, ids_tmp->Jz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxx_flat, ids_tmp->pxx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxy_flat, ids_tmp->pxy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxz_flat, ids_tmp->pxz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyy_flat, ids_tmp->pyy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyz_flat, ids_tmp->pyz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pzz_flat, ids_tmp->pzz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
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
    delete[] ids_tmp;
}