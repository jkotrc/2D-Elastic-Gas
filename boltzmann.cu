#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define mag(x,y) sqrt(x*x+y*y)
#define mag2(x,y) x*x+y*y

#define BLOCK_SIZE 512


__global__ void step_kernel(float *posx, float *posy, float *vx, float *vy, int N, float size, float epsilon, float width, float height) {
			int idx = blockIdx.x*blockDim.x+threadIdx.x;
			//printf("%u, ",idx);

			const float sqrtarg=1+8*idx;
			int j = (int)(sqrt(sqrtarg)+1);
			j/=2;
			int k = idx - (j*(j-1))/2;

			if (j >= N || k >= N){return;}

			float magnitude=mag((posx[k]-posx[j]),(posy[k]-posy[j]));
			if (magnitude <= size) {
					float dot = ((vx[k]-vx[j])*(posx[k]-posx[j]))+((vy[k]-vy[j])*(posy[k]-posy[j]));
					float mg = size*size;
					vx[k] -= (posx[k]-posx[j]) * (dot/mg);
					vy[k] -= (posy[k]-posy[j]) * (dot/mg);
					vx[j] -= (posx[j]-posx[k]) * (dot/mg);
					vy[j] -= (posy[j]-posy[k]) * (dot/mg);

					//when balls get stuck, push one of them just outside the other
					posx[j] -= (size/magnitude-0.8)*(posx[k]-posx[j]);
					posy[j] -= (size/magnitude-0.8)*(posy[k]-posy[j]);
				}

			if (posx[j] >= width-size/2) {
				vx[j] *= -1;
				posx[j] = width-size/2;
			}
			if (posx[j] <= -width+size/2) {
				vx[j] *= -1;
				posx[j] = -width+size/2;
			}
			if (posy[j] >= height-size/2) {
				vy[j] *= -1;
				posy[j] = height-size/2;
			}
			if (posy[j] <= -height+size/2) {
				vy[j] *= -1;
				posy[j] = -height+size/2;
			}
			
			if (posx[k] >= width-size/2) {
				vx[k] *= -1;
				posx[k] = width-size/2;
			}
			if (posx[k] <= -width+size/2) {
				vx[k] *= -1;
				posx[k] = -width+size/2;
			}
			if (posy[k] >= height-size/2) {
				vy[k] *= -1;
				posy[k] = height-size/2;
			}
			if (posy[k] <= -height+size/2) {
				vy[k] *= -1;
				posy[k] = -height+size/2;
			}

			posx[j]+=epsilon*vx[j];
			posy[j]+=epsilon*vy[j];

			posx[k]+=epsilon*vx[k];
			posy[k]+=epsilon*vy[k];
}