#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUAllocation.h"
#include <stdio.h>

#define STREAMS 2  // number of streams for each computation on the GPU

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** deallocate (pinned memory)*/
void particle_deallocate_pinned(struct particles* part)
{
    // deallocate particle variables
    cudaFreeHost(part->x);
    cudaFreeHost(part->y);
    cudaFreeHost(part->z);
    cudaFreeHost(part->u);
    cudaFreeHost(part->v);
    cudaFreeHost(part->w);
    cudaFreeHost(part->q);
}

/** Compute number of batches */
int get_nob(int nop, int batchsize)
{
    return (nop + batchsize - 1) / batchsize;
}

/** Create batches of particles */
int particle_batch_create(struct parameters* param, struct particles* part, struct particles** part_batches)
{
    // Compute number of batches
    int nob = get_nob(part->nop, param->batchsize);

    // Fill one batch at a time with particle data
    // *part_batches = new particles[nob];
    cudaMallocHost((void **) part_batches, sizeof(particles)*nob, cudaHostAllocDefault);

    for (int batch_id=0; batch_id<nob; ++batch_id) {
        // copy structure, pointers will still point to the same memory address
        (*part_batches)[batch_id] = *part;

        //////////////////////////////////////
        // Overwrite relevant scalar values //
        //////////////////////////////////////

        // number of particles
        if (batch_id == nob-1) {
            // Last batch is a special case as it may contain fewer than batchsize particles
            int batch_remainder = part->nop % param->batchsize;
            if (batch_remainder == 0)
                (*part_batches)[batch_id].nop = param->batchsize;
            else
                (*part_batches)[batch_id].nop = batch_remainder;
        }
        else
            (*part_batches)[batch_id].nop = param->batchsize;

        // maximum number of particles
        long npmax = param->batchsize; // I am not really sure if we can just use the batchsize here... 
        (*part_batches)[batch_id].npmax = npmax;
            
        ///////////////////////
        /// Overwrite arrays //
        ///////////////////////

        // Allocate new memory addresses
        // (*part_batches)[batch_id].x = new FPpart[npmax];
        // (*part_batches)[batch_id].y = new FPpart[npmax];
        // (*part_batches)[batch_id].z = new FPpart[npmax];
        // (*part_batches)[batch_id].u = new FPpart[npmax];
        // (*part_batches)[batch_id].v = new FPpart[npmax];
        // (*part_batches)[batch_id].w = new FPpart[npmax];
        // (*part_batches)[batch_id].q = new FPinterp[npmax];
        cudaMallocHost((void **) &((*part_batches)[batch_id].x), sizeof(FPpart)*npmax, cudaHostAllocDefault);
        cudaMallocHost((void **) &((*part_batches)[batch_id].y), sizeof(FPpart)*npmax, cudaHostAllocDefault);
        cudaMallocHost((void **) &((*part_batches)[batch_id].z), sizeof(FPpart)*npmax, cudaHostAllocDefault);
        cudaMallocHost((void **) &((*part_batches)[batch_id].u), sizeof(FPpart)*npmax, cudaHostAllocDefault);
        cudaMallocHost((void **) &((*part_batches)[batch_id].v), sizeof(FPpart)*npmax, cudaHostAllocDefault);
        cudaMallocHost((void **) &((*part_batches)[batch_id].w), sizeof(FPpart)*npmax, cudaHostAllocDefault);
        cudaMallocHost((void **) &((*part_batches)[batch_id].q), sizeof(FPpart)*npmax, cudaHostAllocDefault);

        // Copy the values
        std::copy((part->x)+batch_id*param->batchsize, (part->x)+batch_id*param->batchsize+(*part_batches)[batch_id].nop, (*part_batches)[batch_id].x);
        std::copy((part->y)+batch_id*param->batchsize, (part->y)+batch_id*param->batchsize+(*part_batches)[batch_id].nop, (*part_batches)[batch_id].y);
        std::copy((part->z)+batch_id*param->batchsize, (part->z)+batch_id*param->batchsize+(*part_batches)[batch_id].nop, (*part_batches)[batch_id].z);
        std::copy((part->u)+batch_id*param->batchsize, (part->u)+batch_id*param->batchsize+(*part_batches)[batch_id].nop, (*part_batches)[batch_id].u);
        std::copy((part->v)+batch_id*param->batchsize, (part->v)+batch_id*param->batchsize+(*part_batches)[batch_id].nop, (*part_batches)[batch_id].v);
        std::copy((part->w)+batch_id*param->batchsize, (part->w)+batch_id*param->batchsize+(*part_batches)[batch_id].nop, (*part_batches)[batch_id].w);
        std::copy((part->q)+batch_id*param->batchsize, (part->q)+batch_id*param->batchsize+(*part_batches)[batch_id].nop, (*part_batches)[batch_id].q);
    }

    return nob;
}

/** Deallocate particle batches */
void particle_batch_deallocate(struct particles* part_batches, int nob)
{
    for (int i=0; i<nob; ++i)
    {
        particle_deallocate_pinned(&part_batches[i]);
    }
}

/** particle mover */
__global__
void mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param, long offset, long num_elem)
{
    // get thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // add offset to get global particle ID
    id = offset + id;
        
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        if (id < offset + num_elem){
            xptilde = part->x[id];
            yptilde = part->y[id];
            zptilde = part->z[id];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[id] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[id] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[id] - grd->zStart)*grd->invdz);
                
                // calculate weights
                // xi[0]   = part->x[id] - grd->XN[ix - 1][iy][iz];
                // eta[0]  = part->y[id] - grd->YN[ix][iy - 1][iz];
                // zeta[0] = part->z[id] - grd->ZN[ix][iy][iz - 1];
                xi[0]   = part->x[id] - grd->XN_flat[get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn)];                
                eta[0]  = part->y[id] - grd->YN_flat[get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn)];                
                zeta[0] = part->z[id] - grd->ZN_flat[get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn)];
                // xi[1]   = grd->XN[ix][iy][iz] - part->x[id];
                // eta[1]  = grd->YN[ix][iy][iz] - part->y[id];
                // zeta[1] = grd->ZN[ix][iy][iz] - part->z[id];
                xi[1]   = grd->XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->x[id];                
                eta[1]  = grd->YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->y[id];                
                zeta[1] = grd->ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->z[id];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            // Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            // Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            // Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            // Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            // Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            // Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                            Exl += weight[ii][jj][kk]*field->Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                            Eyl += weight[ii][jj][kk]*field->Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                            Ezl += weight[ii][jj][kk]*field->Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                            Bxl += weight[ii][jj][kk]*field->Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                            Byl += weight[ii][jj][kk]*field->Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                            Bzl += weight[ii][jj][kk]*field->Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[id] + qomdt2*Exl;
                vt= part->v[id] + qomdt2*Eyl;
                wt= part->w[id] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[id] = xptilde + uptilde*dto2;
                part->y[id] = yptilde + vptilde*dto2;
                part->z[id] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[id]= 2.0*uptilde - part->u[id];
            part->v[id]= 2.0*vptilde - part->v[id];
            part->w[id]= 2.0*wptilde - part->w[id];
            part->x[id] = xptilde + uptilde*dt_sub_cycling;
            part->y[id] = yptilde + vptilde*dt_sub_cycling;
            part->z[id] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[id] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[id] = part->x[id] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[id] = -part->u[id];
                    part->x[id] = 2*grd->Lx - part->x[id];
                }
            }
                                                                        
            if (part->x[id] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[id] = part->x[id] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[id] = -part->u[id];
                    part->x[id] = -part->x[id];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[id] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[id] = part->y[id] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[id] = -part->v[id];
                    part->y[id] = 2*grd->Ly - part->y[id];
                }
            }
                                                                        
            if (part->y[id] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[id] = part->y[id] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[id] = -part->v[id];
                    part->y[id] = -part->y[id];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[id] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[id] = part->z[id] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[id] = -part->w[id];
                    part->z[id] = 2*grd->Lz - part->z[id];
                }
            }
                                                                        
            if (part->z[id] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[id] = part->z[id] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[id] = -part->w[id];
                    part->z[id] = -part->z[id];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return; // exit
}

/* launch GPU version of the particle mover */
int mover_PC_gpu_launch(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // Copy EMfield struct to device
    EMfield* field_gpu;
    emfield_move2gpu(field, &field_gpu, grd);

    // Copy grid struct to device
    grid* grd_gpu;
    grid_move2gpu(grd, &grd_gpu);
    
    // Copy parameters struct to device
    parameters* param_gpu;
    cudaMalloc(&param_gpu, sizeof(parameters));
    cudaMemcpy(param_gpu, param, sizeof(parameters), cudaMemcpyHostToDevice);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Divide the particle data in segments and use streams to overlap data transfer and computation //
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Prepare auxiliary variables
    long pps = ceil(part->npmax / STREAMS);  // particles per stream
    long stream_offset[STREAMS];             // array segment offset
    long np_stream[STREAMS];                 // number of particles in stream

    // Create cuda streams and offsets and assign a number of particles to each stream
    cudaStream_t stream[STREAMS];
    for (int s_id=0; s_id<STREAMS; ++s_id)
    {
        cudaStreamCreate(&stream[s_id]);
        
        // Compute offset to specify start of array segments
        stream_offset[s_id] = s_id * pps;

        // Number of particles in stream is either equal to pps or what is left in the last stream
        np_stream[s_id] = std::min(pps, part->nop - stream_offset[s_id]); 
    }

    // Trigger asynchronous copy for each stream
    particles* part_gpu;
    particle_move2gpu(part, &part_gpu, STREAMS, stream, stream_offset, np_stream);

    // Launch kernels for each stream
    for (int s_id=0; s_id<STREAMS; ++s_id)
    {   
        // Call kernel (the third execution configuration parameter is 0 because no shared device memory is allocated)
        mover_PC_gpu<<<(np_stream[s_id]+param->tpb-1)/param->tpb, param->tpb, 0, stream[s_id]>>>(part_gpu, field_gpu, grd_gpu, param_gpu, stream_offset[s_id], np_stream[s_id]);
    }

    // Retrieve data from the device (trigger asynchronous copy)
    particle_move2cpu(part_gpu, part, STREAMS, stream, stream_offset, np_stream);
    
    // wait for GPU operations to finish and destroy streams
    cudaDeviceSynchronize();
    for (int s_id=0; s_id<STREAMS; ++s_id)
    {
        cudaStreamDestroy(stream[s_id]);
    }

    // Free the memory
    particle_deallocate_gpu(part_gpu);
    emfield_deallocate_gpu(field_gpu);
    grid_deallocate_gpu(grd_gpu);
    cudaFree(param_gpu);

    return 0;
}


/** Interpolation Particle --> Grid: This is for species */
__global__
void interpP2G_gpu(struct particles* part, struct interpDensSpecies* ids, struct grid* grd, long offset, long num_elem)
{ 
    // get thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // add offset to get global particle ID
    id = offset + id;

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
     
    if (id < offset + num_elem) {

        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[id] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[id] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[id] - grd->zStart) * grd->invdz));
        
        // distances from node
        // xi[0]   = part->x[id] - grd->XN[ix - 1][iy][iz];
        // eta[0]  = part->y[id] - grd->YN[ix][iy - 1][iz];
        // zeta[0] = part->z[id] - grd->ZN[ix][iy][iz - 1];
        xi[0]   = part->x[id] - grd->XN_flat[get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn)];
        eta[0]  = part->y[id] - grd->YN_flat[get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn)];
        zeta[0] = part->z[id] - grd->ZN_flat[get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn)];
        // xi[1]   = grd->XN[ix][iy][iz] - part->x[id];
        // eta[1]  = grd->YN[ix][iy][iz] - part->y[id];
        // zeta[1] = grd->ZN[ix][iy][iz] - part->z[id];
        xi[1]   = grd->XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->x[id];
        eta[1]  = grd->YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->y[id];
        zeta[1] = grd->ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->z[id];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[id] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    // ids->rhon_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += weight[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->rhon_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), weight[ii][jj][kk] * grd->invVOL);
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[id] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    // ids->Jx_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += temp[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->Jx_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), temp[ii][jj][kk] * grd->invVOL);
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[id] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    // ids->Jy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += temp[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->Jy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), temp[ii][jj][kk] * grd->invVOL);
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[id] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    // ids->Jz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += temp[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->Jz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), temp[ii][jj][kk] * grd->invVOL);
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[id] * part->u[id] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    // ids->pxx_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += temp[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->pxx_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), temp[ii][jj][kk] * grd->invVOL);
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[id] * part->v[id] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    // ids->pxy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += temp[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->pxy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), temp[ii][jj][kk] * grd->invVOL);
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[id] * part->w[id] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    // ids->pxz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += temp[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->pxz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), temp[ii][jj][kk] * grd->invVOL);
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[id] * part->v[id] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    // ids->pyy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += temp[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->pyy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), temp[ii][jj][kk] * grd->invVOL);
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[id] * part->w[id] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    // ids->pyz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += temp[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->pyz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), temp[ii][jj][kk] * grd->invVOL);
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[id] * part->w[id] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    // ids->pzz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)] += temp[ii][jj][kk] * grd->invVOL;
                    atomicAdd(&(ids->pzz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)]), temp[ii][jj][kk] * grd->invVOL);
    
    }
   
}

/* launch GPU version of the P2G interpolation */
int interpP2G_gpu_launch(struct particles* part, struct interpDensSpecies* ids, struct grid* grd, struct parameters* param)
{
    // Copy interpDensSpecies struct to device
    interpDensSpecies* ids_gpu;
    ids_move2gpu(ids, &ids_gpu, grd);

    // Copy grid struct to device
    grid* grd_gpu;
    grid_move2gpu(grd, &grd_gpu);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Divide the particle data in segments and use streams to overlap data transfer and computation //
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Prepare auxiliary variables
    long pps = ceil(part->npmax / STREAMS);  // particles per stream
    long stream_offset[STREAMS];             // array segment offset
    long np_stream[STREAMS];                 // number of particles in stream

    // Create cuda streams and offsets and assign a number of particles to each stream
    cudaStream_t stream[STREAMS];
    for (int s_id=0; s_id<STREAMS; ++s_id)
    {
        cudaStreamCreate(&stream[s_id]);
        
        // Compute offset to specify start of array segments
        stream_offset[s_id] = s_id * pps;

        // Number of particles in stream is either equal to pps or what is left in the last stream
        np_stream[s_id] = std::min(pps, part->nop - stream_offset[s_id]); 
    }

    // Trigger asynchronous copy for each stream
    particles* part_gpu;
    particle_move2gpu(part, &part_gpu, STREAMS, stream, stream_offset, np_stream);

    // Launch kernels for each stream
    for (int s_id=0; s_id<STREAMS; ++s_id)
    {
        // Call kernel (the third execution configuration parameter is 0 because no shared device memory is be allocated)
        interpP2G_gpu<<<(np_stream[s_id]+param->tpb-1)/param->tpb, param->tpb, 0, stream[s_id]>>>(part_gpu, ids_gpu, grd_gpu, stream_offset[s_id], np_stream[s_id]);
    }

    // wait for GPU operations to finish and destroy streams
    cudaDeviceSynchronize();
    for (int s_id=0; s_id<STREAMS; ++s_id)
    {
        cudaStreamDestroy(stream[s_id]);
    }

    // Retrieve data from the device
    ids_move2cpu(ids_gpu, ids, grd);

    // Free the memory
    particle_deallocate_gpu(part_gpu);
    ids_deallocate_gpu(ids_gpu);
    grid_deallocate_gpu(grd_gpu);

    return 0;
}
