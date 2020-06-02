/*
                                IMPORTANT INFORMATION
NOTE:block size should be a multiple of the warp size (32)

===Attributes for device 0
ASYNC_ENGINE_COUNT:2
CAN_MAP_HOST_MEMORY:1
CLOCK_RATE:1455000
COMPUTE_CAPABILITY_MAJOR:6
COMPUTE_CAPABILITY_MINOR:1
COMPUTE_MODE:DEFAULT
CONCURRENT_KERNELS:1
ECC_ENABLED:0
GLOBAL_L1_CACHE_SUPPORTED:1
GLOBAL_MEMORY_BUS_WIDTH:128
GPU_OVERLAP:1
INTEGRATED:0
KERNEL_EXEC_TIMEOUT:1
L2_CACHE_SIZE:1048576
LOCAL_L1_CACHE_SUPPORTED:1
MANAGED_MEMORY:1

MAX_BLOCK_DIM_X:1024
MAX_BLOCK_DIM_Y:1024
MAX_BLOCK_DIM_Z:64

MAX_GRID_DIM_X:2147483647
MAX_GRID_DIM_Y:65535
MAX_GRID_DIM_Z:65535
MAX_PITCH:2147483647
MAX_REGISTERS_PER_BLOCK:65536
MAX_REGISTERS_PER_MULTIPROCESSOR:65536
MAX_SHARED_MEMORY_PER_BLOCK:49152
MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:98304
MAX_THREADS_PER_BLOCK:1024
MAX_THREADS_PER_MULTIPROCESSOR:2048
MEMORY_CLOCK_RATE:3504000
MULTIPROCESSOR_COUNT:6
MULTI_GPU_BOARD:0
MULTI_GPU_BOARD_GROUP_ID:0
PCI_BUS_ID:28
PCI_DEVICE_ID:0
PCI_DOMAIN_ID:0
STREAM_PRIORITIES_SUPPORTED:1
SURFACE_ALIGNMENT:512
TCC_DRIVER:0
TEXTURE_ALIGNMENT:512
TEXTURE_PITCH_ALIGNMENT:32
TOTAL_CONSTANT_MEMORY:65536
UNIFIED_ADDRESSING:1
WARP_SIZE:32
*/

#include <cuda.h>
#include <stdio.h>
//#include <math.h>



/*__global__ void printIndex(int* rando, int N) {

    //printf("with grid dimensions: %u\n", gridDim.x);

    #define STRIDE 3
    #define OFFSET 0


    int n_elem_per_thread  = N / (gridDim.x * blockDim.x);
    int block_start_idx = n_elem_per_thread * blockIdx.x * blockDim.x;
    int thread_start_idx = block_start_idx
            + (threadIdx.x / STRIDE) * n_elem_per_thread * STRIDE
            + ((threadIdx.x + OFFSET) % STRIDE);

    int thread_end_idx = thread_start_idx + n_elem_per_thread * STRIDE;
    if(thread_end_idx > N) thread_end_idx = N;


    for (int idx = thread_start_idx; idx < thread_end_idx; idx+=STRIDE) {
        //printf("IDX %u; BLOCK %u; Thread %u goes from %u to %u\n",idx, blockIdx.x, threadIdx.x, thread_start_idx, thread_end_idx);
        //printf("loop must run %u times\n", (thread_end_idx-thread_start_idx)/3);
        rando[idx]=0;
        printf("IDX %u; BLOCK %u; BLOCKSTART %u; THREAD %u; %u PER THREAD\n", idx, blockIdx.x, block_start_idx, threadIdx.x, n_elem_per_thread);
    }


}*/

__global__ void add1(int* rando, int N) {
    #define STRIDE 1
    #define OFFSET 0
    int n_elem_per_thread  = N / (gridDim.x * blockDim.x);
    int block_start_idx = n_elem_per_thread * blockIdx.x * blockDim.x;
    int thread_start_idx = block_start_idx
            + (threadIdx.x / STRIDE) * n_elem_per_thread * STRIDE
            + ((threadIdx.x + OFFSET) % STRIDE);

    int thread_end_idx = thread_start_idx + n_elem_per_thread * STRIDE;
    if(thread_end_idx > N) thread_end_idx = N;
    for (int idx = thread_start_idx; idx < thread_end_idx; idx+=STRIDE) {
        rando[idx]+=1;
    }
}