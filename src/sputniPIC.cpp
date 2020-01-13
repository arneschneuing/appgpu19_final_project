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
    double iGPU, iCPU, eGPU= 0.0, eCPU = 0.0;
    float elapsed = 0.0;
    
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
    emfield_move2gpu(&field, &field_gpu, &grd);
    grid_move2gpu(&grd, &grd_gpu);
    cudaMalloc(&param_gpu, sizeof(parameters));
    cudaMemcpy(param_gpu, &param, sizeof(parameters), cudaMemcpyHostToDevice);
    for (int is=0; is < param.ns; is++)
    {
        ids_tmp[is].species_ID = ids[is].species_ID;  // copy species ID for the sake of completeness
        ids_allocate_gpu(&ids_tmp[is], &ids_gpu[is], &grd);
        // update relevant scalar values in temporary structure (scalar values will be copied in particle_allocate_gpu())
        part_tmp[is].qom = part_batches[is][0].qom;
        part_tmp[is].n_sub_cycles = part_batches[is][0].n_sub_cycles;
        part_tmp[is].NiterMover = part_batches[is][0].NiterMover;
        particle_allocate_gpu(&part_tmp[is], &part_gpu[is], batchsize[is]);
    }  

    /////////////////////////////////////////////////////////////////////////////////
    // Prepare streams for concurrency (overlapping data transfer and computation) //
    /////////////////////////////////////////////////////////////////////////////////
    long np_stream_tot[param.n_streams];  // total number of particles in stream
    long np_stream_free;  // remaining number of particles in stream (temporary variable to fill one stream with multiple species)
    long np_stream[param.n_streams][param.ns];  // number of particles in stream for each species
    long offset_stream[param.n_streams][param.ns];  // particle offset for each species
    long nop_remaining;  // remaining particles of one species (helper variable)

    // Calculate total number of particles per batch (all species)
    long np_tot = 0;
    for (int is=0; is < param.ns; is++)
        np_tot += batchsize[is];

    // Create cuda streams and offsets and assign a number of particles to each stream
    cudaStream_t stream[param.n_streams];
    long pps = (np_tot+param.n_streams-1) / param.n_streams;  // particles per stream
    for (int s_id=0; s_id<param.n_streams; ++s_id)
    {
        cudaStreamCreate(&(stream[s_id]));

        // Number of particles in stream is either equal to pps or what is left in the last stream
        if (s_id == param.n_streams-1)
            np_stream_tot[s_id] = np_tot - pps*s_id;
        else
            np_stream_tot[s_id] = pps; 
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
        iGPU = cpuSecond(); // start timer
        for (int is=0; is < param.ns; is++)
            ids_move2gpu(&ids[is], &ids_tmp[is], &grd);
        
        for (int ib=0; ib<param.nob; ++ib)
        {
            std::cout << "batch index: " << ib << std::endl;

            for (int s_id=0; s_id<param.n_streams; ++s_id)
            {         
                // Reset helper variable
                np_stream_free = np_stream_tot[s_id];

                // Compute offset and number of particles in stream for each species
                for (int is=0; is < param.ns; is++)
                {
                    if (s_id == 0)
                        offset_stream[s_id][is] = 0;
                    else
                        offset_stream[s_id][is] = offset_stream[s_id-1][is] + np_stream[s_id-1][is];  // new offset = old offset + #particles in last stream
                    nop_remaining = part_batches[is][ib].nop - offset_stream[s_id][is];
                    if (nop_remaining <= np_stream_free)  // all remaining particles of one species fit into the stream
                        np_stream[s_id][is] = nop_remaining;                        
                    else
                        np_stream[s_id][is] = np_stream_free;
                    np_stream_free -= np_stream[s_id][is];            
                }
            }

            // move particle data to GPU
            for (int s_id=0; s_id<param.n_streams; ++s_id)
            {
                // data movement can be reduced if all particles fit in GPU memory (by transferring only in the first cycle)
                if (param.nob > 1 || cycle == 1)
                {
                    // Move new batch of particles to GPU
                    for (int is=0; is < param.ns; is++)
                    {
                        if (np_stream[s_id][is] > 0)
                            particle_move2gpu(&part_batches[is][ib], &part_tmp[is], &part_gpu[is], stream[s_id], offset_stream[s_id][is], np_stream[s_id][is]);
                    }         
                }
            }

            // Launch combined kernel for particle updates and grid interpolation
            for (int s_id=0; s_id<param.n_streams; ++s_id)
            {  
                for (int is=0; is < param.ns; is++)
                {
                    if (np_stream[is] > 0)  // only launch kernel if current stream contains particles of this species
                        simple_pic_step_launch(part_gpu[is], field_gpu, ids_gpu[is], grd_gpu, param_gpu, np_stream[is], param.tpb, stream[s_id], offset_stream[is]);
                }
            }

            // Retrieve particle mover result
            for (int s_id=0; s_id<param.n_streams; ++s_id)
            {
                if (param.nob > 1)
                {
                    for (int is=0; is < param.ns; is++)
                    {
                        if (np_stream[s_id][is] > 0)
                            particle_move2cpu(&part_tmp[is], &part_batches[is][ib], stream[s_id], offset_stream[s_id][is], np_stream[s_id][is]);
                    }
                }            
            }
            
        }
        cudaDeviceSynchronize();
        // Retrieve data from the device
        for (int is=0; is < param.ns; is++)
            ids_move2cpu(&ids_tmp[is], &ids[is], &grd);
        eGPU += (cpuSecond() - iGPU); // stop timer

        iCPU = cpuSecond();
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        eCPU += (cpuSecond() - iCPU);
        
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
    cudaFreeHost(ids);
    cudaFreeHost(part_batches);

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
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   GPU Time / Cycle     (s) = " << eGPU/param.ncycles << std::endl;
    std::cout << "   CPU Time / Cycle     (s) = " << eCPU/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


