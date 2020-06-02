#include <cuda.h>
#include <stdio.h>

#define mag(x,y) sqrt(x*x+y*y)
#define mag2(x,y) x*x+y*y

__global__ void collisioncheck(float *posx, float *posy, float *vx, float *vy, int N, float size, float width, float height) {
            /*
                Check for collisions; this is a loop over all pairs of balls. There are (N/2)(N-1) many
                Each thread computes the collision between two pairs of balls with indices (j,k1) and (j,k2)
                As such, N(N/2) threads are needed in total.

                With normal serial computation, this would be the same as two loops for j=0;j < k1 and j=0; j < k2, where
                k1=0...N/2
                k2=N...N/2
            */

            //this indexing method works for the most part but skips the balls sometimes, making this some sort of semi-ideal gas. Needs debugging
            const int blocks_per_loop = ( (N-1) -1+blockDim.x)/(blockDim.x);  //one loop = for j=0;j < k1 and j=0; j < k2, i.e. (N-1) iterations
            //j counter resets every blocks_per_loop-th block
            const int k1 = blockIdx.x/blocks_per_loop;
            const int k2 = N-1-k1;

            int j;
            const int remainder = blockIdx.x % blocks_per_loop;
            if (remainder == 0) {
                j = threadIdx.x;
            } else {
                j = threadIdx.x+remainder*blockDim.x;
            }
            if (j > N-1) {
                j=N-1;
            }
                if (j < k1) {
                    float magnitude=mag((posx[k1]-posx[j]),(posy[k1]-posy[j]));
                    if (magnitude <= size) {
                            float dot = ((vx[k1]-vx[j])*(posx[k1]-posx[j]))+((vy[k1]-vy[j])*(posy[k1]-posy[j]));
                            float mg = size*size;
                            vx[k1] -= (posx[k1]-posx[j]) * (dot/mg);
                            vy[k1] -= (posy[k1]-posy[j]) * (dot/mg);
                            vx[j] -= (posx[j]-posx[k1]) * (dot/mg);
                            vy[j] -= (posy[j]-posy[k1]) * (dot/mg);

                            //when balls get stuck, push one of them just outside the other
                            posx[j] -= (size/magnitude-1)*(posx[k1]-posx[j]);
                            posy[j] -= (size/magnitude-1)*(posy[k1]-posy[j]);
                        }
                }

                if (j < k2) {
                    float magnitude=mag((posx[k2]-posx[j]),(posy[k2]-posy[j]));
                    if (magnitude <= size) {
                            float dot = ((vx[k2]-vx[j])*(posx[k2]-posx[j]))+((vy[k2]-vy[j])*(posy[k2]-posy[j]));
                            float mg = size*size;
                            vx[k2] -= (posx[k2]-posx[j]) * (dot/mg);
                            vy[k2] -= (posy[k2]-posy[j]) * (dot/mg);
                            vx[j] -= (posx[j]-posx[k2]) * (dot/mg);
                            vy[j] -= (posy[j]-posy[k2]) * (dot/mg);

                            //when balls get stuck, push one of them just outside the other
                            posx[j] -= (size/magnitude-1)*(posx[k2]-posx[j]);
                            posy[j] -= (size/magnitude-1)*(posy[k2]-posy[j]);
                        }
                }
}


__device__ float round_to_epsilon(float value, float epsilon) {
    const int multiple = round(value/epsilon);
    return multiple*epsilon;
}


__global__ void step(float *posx, float *posy, float *vx, float *vy, int N, float size, float epsilon, float width, float height) {

    const int block_start_idx = blockIdx.x*blockDim.x;
    int idx = block_start_idx+threadIdx.x;
    if (idx > N) {
        idx=N;
    }

    //collisions with the wall
    if (posx[idx] >= width-size/2) {
        vx[idx] *= -1;
        posx[idx] = width-size/2;
    }
    if (posx[idx] <= -width+size/2) {
        vx[idx] *= -1;
        posx[idx] = -width+size/2;
    }
    if (posy[idx] >= height-size/2) {
        vy[idx] *= -1;
        posy[idx] = height-size/2;
    }
    if (posy[idx] <= -height+size/2) {
        vy[idx] *= -1;
        posy[idx] = -height+size/2;
    }

    //do step
    posx[idx]+=epsilon*vx[idx];
    posy[idx]+=epsilon*vy[idx];

    //round the position and velocity to be a multiple of epsilon, so that the floating point errors don't start adding up
    posx[idx]=round_to_epsilon(posx[idx],epsilon);
    posy[idx]=round_to_epsilon(posy[idx],epsilon);
    vx[idx]=round_to_epsilon(vx[idx],epsilon);
    vy[idx]=round_to_epsilon(vy[idx],epsilon);
}

