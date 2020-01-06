/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#include "GPUAllocation.h"


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iInterp, eMover = 0.0, eInterp= 0.0;
    float elapsed = 0.0;

    // Use CUDA event timers
    cudaEvent_t mover_start, mover_stop, interp_start, interp_stop;
    cudaEventCreate(&mover_start);
    cudaEventCreate(&mover_stop);
    cudaEventCreate(&interp_start);
    cudaEventCreate(&interp_stop);
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids;  // interpDensSpecies *ids = new interpDensSpecies[param.ns];
    cudaMallocHost((void **) &ids, sizeof(interpDensSpecies)*param.ns, cudaHostAllocDefault);
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);

    // Create mini-batches
    particles** part_batches;  // particles **part_batches = new particles*[param.ns];
    cudaMallocHost((void **) &part_batches, sizeof(particles*)*param.ns, cudaHostAllocDefault);
    int batchsize[param.ns];  // batchsize might differ for different species
    for (int is=0; is < param.ns; is++){
        batchsize[is] = particle_batch_create(&param, &part[is], &part_batches[is]);
    }

    // Deallocate (un-batched) particles
    for (int is=0; is < param.ns; is++){
        particle_deallocate(&part[is]);
    }

    /////////////////////////
    // GPU data management //
    /////////////////////////
    // Declare GPU variables
    particles* part_gpu[param.ns];
    particles *part_tmp = new particles[param.ns];  // container to make device pointers accessible from the host
    EMfield* field_gpu;
    grid* grd_gpu;
    parameters* param_gpu;
    interpDensSpecies* ids_gpu[param.ns];
    interpDensSpecies *ids_tmp = new interpDensSpecies[param.ns];  // container to make device pointers accessible from the host
    
    // Allocate memory and move data to the GPU
    for (int is=0; is < param.ns; is++)
    {
        // update relevant scalar values in temporary structure (scalar values will be copied in particle_allocate_gpu())
        part_tmp[is].qom = part_batches[is][0].qom;
        part_tmp[is].n_sub_cycles = part_batches[is][0].n_sub_cycles;
        part_tmp[is].NiterMover = part_batches[is][0].NiterMover;
        particle_allocate_gpu(&part_tmp[is], &part_gpu[is], batchsize[is]);
        ids_tmp[is].species_ID = ids[is].species_ID;  // copy species ID for the sake of completeness
        ids_allocate_gpu(&ids_tmp[is], &ids_gpu[is], &grd);
    }  
    emfield_move2gpu(&field, &field_gpu, &grd);
    grid_move2gpu(&grd, &grd_gpu);
    cudaMalloc(&param_gpu, sizeof(parameters));
    cudaMemcpy(param_gpu, &param, sizeof(parameters), cudaMemcpyHostToDevice);

    /////////////////////////////////////////////////////////////////////////////////
    // Prepare streams for concurrency (overlapping data transfer and computation) //
    /////////////////////////////////////////////////////////////////////////////////
    long np_stream_tot[param.n_streams];  // total number of particles in stream
    long np_stream_free;  // remaining number of particles in stream (temporary variable to fill one stream with multiple species)
    long np_stream[param.ns];  // number of particles in current stream for each species
    long offset_stream[param.ns];  // particle offset for each species
    long nop_remaining;  // remaining particles of one species (helper variable)

    // Calculate total number of particles (all species)
    long np_tot = 0;
    for (int is=0; is < param.ns; is++)
        np_tot += param.np[is];

    // Create cuda streams and offsets and assign a number of particles to each stream
    cudaStream_t stream[param.n_streams];
    long pps = (np_tot+param.n_streams-1) / param.n_streams;  // particles per stream
    for (int s_id=0; s_id<param.n_streams; ++s_id)
    {
        cudaStreamCreate(&stream[s_id]);

        // Number of particles in stream is either equal to pps or what is left in the last stream
        if (s_id == param.n_streams-1)
            np_stream_tot[s_id] = np_tot - pps*s_id;
        else
            np_stream_tot[s_id] = pps; 
    }

    // Use events to synchronize streams whithout blocking the host
    cudaEvent_t mover_sync[param.n_streams];
    cudaEvent_t interp_sync[param.n_streams];
    for (int s_id=0; s_id<param.n_streams; ++s_id)
    {
        cudaEventCreate(&mover_sync[s_id]);
        cudaEventCreate(&interp_sync[s_id]);
    }
    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);

        // Copy ids to device
        iInterp = cpuSecond(); // start timer for the first part of the interpolation step
        for (int is=0; is < param.ns; is++)
            ids_move2gpu(&ids[is], &ids_tmp[is], &grd);
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        for (int ib=0; ib<param.nob; ++ib)
        {
            std::cout << "batch index: " << ib << std::endl;

            // Reset helper variables
            for (int is=0; is < param.ns; is++)
            {
                offset_stream[is] = 0;
                np_stream[is] = 0;
            }

            for (int s_id=0; s_id<param.n_streams; ++s_id)
            {         
                // Reset helper variable
                np_stream_free = np_stream_tot[s_id];

                // Compute offset and number of particles in stream for each species
                for (int is=0; is < param.ns; is++)
                {
                    offset_stream[is] += np_stream[is];  // new offset = old offset + #particles in last stream
                    nop_remaining = part_batches[is][ib].nop - offset_stream[is];
                    if (nop_remaining <= np_stream_free)  // all remaining particles of one species fit into the stream
                        np_stream[is] = nop_remaining;                        
                    else
                        np_stream[is] = np_stream_free;
                    np_stream_free -= np_stream[is];               
                }


                // implicit mover
                if (s_id == 0)  // start timer when the first stream is processed
                    cudaEventRecord(mover_start, stream[s_id]);

                // data movement can be reduced if all particles fit in GPU memory (by transferring only in the first cycle)
                if (param.nob > 1 || cycle == 1)
                {
                    // Move new batch of particles to GPU
                    for (int is=0; is < param.ns; is++)
                    {
                        if (np_stream[is] > 0)
                            particle_move2gpu(&part_batches[is][ib], &part_tmp[is], &part_gpu[is], stream[s_id], offset_stream[is], np_stream[is]);
                    }         
                }
                for (int is=0; is < param.ns; is++)
                {
                    if (np_stream[is] > 0)  // only launch kernel if current stream contains particles of this species
                        mover_PC_gpu_launch(part_gpu[is], field_gpu, grd_gpu, param_gpu, np_stream[is], param.tpb, stream[s_id], offset_stream[is]);
                }

                // make sure all particles have been moved before continuing
                cudaEventRecord(mover_sync[s_id], stream[s_id]);
                cudaStreamWaitEvent(stream[s_id], mover_sync[s_id]);


                if (param.nob > 1)
                {
                    // Retrieve particle mover result
                    for (int is=0; is < param.ns; is++)
                    {
                        if (np_stream[is] > 0)
                            particle_move2cpu(&part_tmp[is], &part_batches[is][ib], stream[s_id], offset_stream[is], np_stream[is]);
                    }
                }
                if (s_id == param.n_streams-1)  // stop timer when the last stream finished
                    cudaEventRecord(mover_stop, stream[s_id]);
            
            
                // interpolation particle to grid
                if (s_id == 0)  // start timer when the first stream is processed
                    cudaEventRecord(interp_start, stream[s_id]);

                // interpolate species
                for (int is=0; is < param.ns; is++)
                {
                    if (np_stream[is] > 0)  // only launch kernel if current stream contains particles of this species
                        interpP2G_gpu_launch(part_gpu[is], ids_gpu[is], grd_gpu, np_stream[is], param.tpb, stream[s_id], offset_stream[is]);
                } 
                // block further execution until all particles (in stream) have been interpolated to the grid
                cudaEventRecord(interp_sync[s_id], stream[s_id]);
                cudaStreamWaitEvent(stream[s_id], interp_sync[s_id]);

                if (s_id == param.n_streams-1)  // stop timer when the last stream finished
                    cudaEventRecord(interp_stop, stream[s_id]);
            }
            // Update timers
            cudaEventSynchronize(mover_stop);
            cudaEventElapsedTime(&elapsed, mover_start, mover_stop);
            eMover += elapsed;
            cudaEventSynchronize(interp_stop);
            cudaEventElapsedTime(&elapsed, interp_start, interp_stop);
            eInterp += elapsed;
        }
        iInterp = cpuSecond(); // start timer for the rest of interpolation step
        // Retrieve data from the device
        for (int is=0; is < param.ns; is++)
            ids_move2cpu(&ids_tmp[is], &ids[is], &grd);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        
        
    
    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_batch_deallocate(part_batches[is], param.nob);
    }

    // Free GPU memory
    for (int is=0; is < param.ns; is++)
    {
        particle_deallocate_gpu(part_gpu[is]);
        ids_deallocate_gpu(ids_gpu[is]);
    }
    emfield_deallocate_gpu(field_gpu);
    grid_deallocate_gpu(grd_gpu);
    cudaFree(param_gpu);

    // destroy cuda streams
    for (int s_id=0; s_id<param.n_streams; ++s_id)
        cudaStreamDestroy(stream[s_id]); 

    // destroy event handles
    cudaEventDestroy(mover_start);
    cudaEventDestroy(mover_stop);
    cudaEventDestroy(interp_start);
    cudaEventDestroy(interp_stop);
    for (int s_id=0; s_id<param.n_streams; ++s_id)
    {
        cudaEventDestroy(mover_sync[s_id]);
        cudaEventDestroy(interp_sync[s_id]);
    }
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


