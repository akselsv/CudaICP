#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <limits>
#include <ctime>
#include <iostream>

#include "kernel.h"

#define ROW 3 //the dimension
#define COL 217088 //number of points

__device__ const int numPoints = 217088;
__device__ const int blockSize = 512;

int threads = blockSize;

__global__ void kernelNNStructured
(float *pts1, float*pts2, float *distances, int *closest, int area) {
	// This implementation exploits the matrix structure of a depth image from a 3D camera and perfroms a limited NN search for each point. 
	// For a query point q with row coordinate a and column coordinate b in matrix pts1, this algorithm searches through the points in the range a-n:a+n b-m:b+n in matrix pts2
	// This assumes small increments in the movement between the two consecutive depth images pts1 and pts2
	// This implementation is hard-coded for a 512x424 depth image

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	float distToClosest = 999999;
	int closestIndex = -1;

	if (idx < numPoints) {
		int a = idx / 512; //this row
		int b = idx % 512; //this col
		for (int i = -area; i <= area; i++) {
			for (int j = -area; j <= area; j++) {
				if (a + i >= 0 && a + i < 424 && b + j >= 0 && b + j < 512 && (i*i + j*j) <= (area+1)*(area+1)) { //this check makes sure we are within the limits of the 512x424 matrix and that we do a radius search instead of a square search
					int this_ind = (a + i) * 512 + b + j;
					float dist = (pts1[idx * 3 + 0] - pts2[this_ind * 3 + 0])*(pts1[idx * 3 + 0] - pts2[this_ind * 3 + 0]) +
						(pts1[idx * 3 + 1] - pts2[this_ind * 3 + 1])*(pts1[idx * 3 + 1] - pts2[this_ind * 3 + 1]) +
						(pts1[idx * 3 + 2] - pts2[this_ind * 3 + 2])*(pts1[idx * 3 + 2] - pts2[this_ind * 3 + 2]);
					if (dist < distToClosest) {
						distToClosest = dist;
						closestIndex = this_ind;
					}
				}
			}
		}
		if (distToClosest > 0.010) { //treshold test
			closestIndex = -1;
		}
		closest[idx] = closestIndex;
		distances[idx] = distToClosest;
	}
}

__global__ void kernelNN2
(float *pts1, float*pts2, float *distances, int *closest) {
	// This implementation uses "blocking". It copies chuncks of data from the global 
	//memory to the shared memory and performs calculation directly on shared memory instead.
	//The speed up is with a factor of 2 compared to working only on global memory
	__shared__ float sharedPoints[3 * blockSize];

	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	float distToClosest = 999999;
	int closestIndex = -1;

	if (idx < numPoints) {
		for (int currentBlockOfPoints = 0; currentBlockOfPoints < gridDim.x; currentBlockOfPoints++) {
			if (threadIdx.x + currentBlockOfPoints*blockSize < numPoints) {
				sharedPoints[threadIdx.x * 3 + 0] = pts2[(threadIdx.x + currentBlockOfPoints*blockSize) * 3 + 0];
				sharedPoints[threadIdx.x * 3 + 1] = pts2[(threadIdx.x + currentBlockOfPoints*blockSize) * 3 + 1];
				sharedPoints[threadIdx.x * 3 + 2] = pts2[(threadIdx.x + currentBlockOfPoints*blockSize) * 3 + 2];

				__syncthreads(); // This call is essential, as it ensures that for the current block, all data have been copies to shared memory

				for (int i = 0; i < blockSize; i++) {
					float dist = (pts1[idx * 3 + 0] - sharedPoints[i * 3 + 0])*(pts1[idx * 3 + 0] - sharedPoints[i * 3 + 0]) +
						(pts1[idx * 3 + 1] - sharedPoints[i * 3 + 1])*(pts1[idx * 3 + 1] - sharedPoints[i * 3 + 1]) +
						(pts1[idx * 3 + 2] - sharedPoints[i * 3 + 2])*(pts1[idx * 3 + 2] - sharedPoints[i * 3 + 2]);

					if (dist < distToClosest) {
						distToClosest = dist;
						closestIndex = i + currentBlockOfPoints*blockDim.x;
					}
				}
			}
			__syncthreads();
			//if (distToClosest > 0.010) { //treshold test
				//closestIndex = -1;
			//}
			closest[idx] = closestIndex;
			distances[idx] = distToClosest;
		}
	}
}

__global__ void edgeDetect(float *pts, int *indices, int num_regions) {// int *edge_indices, float edge_tresh, int num_regions, int region_size) {
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	//Only looking at Z values for kinect camera to detect edges. Exploit that point come organized in a matrix structure directly from the sensor. Looking at 4x4 regions in the depth matrix

	//matrix comes as 512x424 array. Divide by 128 in the x direction and 106 in y direction gives 4x4 regions. This give a total of 13568 regions.

	//Each thread handles 1 region

	//Using incremental calculation of standard deviation for computation saving http://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation

	if (idx < num_regions) {
		int pos = (int)(idx / 128) * 512 * 4 + (idx % 128) * 4;
		float prev_mean = 0;
		float this_var = 0;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				float n = i * 4 + j + 1;
				float this_point = pts[(pos + j + i * 512) * 3 + 2];
				if (n > 1) {
					this_var = ((n - 2) / (n - 1))*this_var + (1 / n)*(this_point - prev_mean)*(this_point - prev_mean);
				}
				prev_mean = (this_point + (n - 1)*prev_mean) / n;
			}
		}//Need to iterate through all points 1 for finding the variance 
		//and 1 for setting if a point is on an edge or not
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				if (this_var > 0.000144 || !isfinite(this_var)) { //25
					indices[(pos + j + i * 512)] = 1;
				}
				else {
					indices[(pos + j + i * 512)] = 0;
				}
			}
		}
	}
}

__global__ void rmEdgesFromCorr(int *closest, int *edges1, int *edges2) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numPoints) {
		if (closest[idx] != -1) {
			if (edges1[idx] == 1 || edges2[closest[idx]] == 1) {
				closest[idx] = -1;
			}
		}
	}
}

__global__ void kernelNN
(float *pts1, float *pts2, float *distances, int *closest)
{
	// det er 1 tråd for hvert punkt i pts1
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < COL) {
		float min_distance = 9999999;
		int closest_idx = -1;

		for (int i = 0; i < COL; i++) {
			float dist = (pts1[idx * 3 + 0] - pts2[i * 3 + 0]) * (pts1[idx * 3 + 0] - pts2[i * 3 + 0]) +
				(pts1[idx * 3 + 1] - pts2[i * 3 + 1]) * (pts1[idx * 3 + 1] - pts2[i * 3 + 1]) +
				(pts1[idx * 3 + 2] - pts2[i * 3 + 2]) * (pts1[idx * 3 + 2] - pts2[i * 3 + 2]);

			if (dist < min_distance) {
				min_distance = dist;
				closest_idx = i;
			}
		}
		closest[idx] = closest_idx;
		distances[idx] = min_distance;
	}
}

__global__ void testMindist
(float *pts1, float *pts2, float *distarr, float *mindist) {
	unsigned int idx = threadIdx.x;
	float dist;
	float curr_dist = 99999;
	__shared__ float min_dist;
	if (idx == 0) {
		min_dist = 99999;
		//mindist[0] = 100;
	}

	__syncthreads();
	if (idx < 1024) {
		for (int i = 0; i < 1024; i++) {
			dist = (pts1[idx * 3 + 0] - pts2[i * 3 + 0])*(pts1[idx * 3 + 0] - pts2[i * 3 + 0]) +
				(pts1[idx * 3 + 1] - pts2[i * 3 + 1])*(pts1[idx * 3 + 1] - pts2[i * 3 + 1]) +
				(pts1[idx * 3 + 2] - pts2[i * 3 + 2])*(pts1[idx * 3 + 2] - pts2[i * 3 + 2]);
			if (dist < curr_dist) {
				curr_dist = dist;
			}
		}
		distarr[idx] = curr_dist;
		if (curr_dist < min_dist && curr_dist != 0) {
			min_dist = curr_dist;
			mindist[0] = min_dist;

		}

	}
}

//__device__ void calcDist(float *pt, float *pts, float *dist)

__global__ void kernelRmDup
(int *closest, float *distances)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;

	if (idx < COL) {
		if (closest[idx] != -1) {
			for (int i = 0; i < COL; i++) {
				if (closest[i] != -1) {
					if (closest[i] == closest[idx] && distances[i] < distances[idx]) {
						closest[idx] = -1;
						break;
					}
				}
			}
		}

	}
}

__global__ void kernelCreateSVD
(float *pts1, float *pts2, float *closest, float* c1, float *c2)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;

	if (idx < COL) {

	}
}



void computeCentroids(float *pt1, float *pt2, float *c1, float *c2) {
	//Computing centroids is a highly sequential task, thus this is done on the host
	int it1 = 0;

	int it2 = 0;
	for (int i = 0; i < COL; i++) {
		if (pt1[i * 3] != -std::numeric_limits<float>::infinity()) {
			c1[0] += pt1[i * 3 + 0];
			c1[1] += pt1[i * 3 + 1];
			c1[2] += pt1[i * 3 + 2];
			it1++;
		}
		if (pt2[i * 3] != -std::numeric_limits<float>::infinity()) {
			c2[0] += pt2[i * 3 + 0];
			c2[1] += pt2[i * 3 + 1];
			c2[2] += pt2[i * 3 + 2];
			it2++;
		}
	}
	for (int i = 0; i < ROW; i++) {
		c1[i] = c1[i] / it1;
		c2[i] = c2[i] / it2;
	}
}

float run_kernel
(float *pts1, float *pts2, float* distances, int *closest, int *edges1, int *edges2)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 dimBlock(16, 1, 1); //threads per block
	dim3 dimGrid(ceil((float)COL / dimBlock.x)); //number of blocks
	printf("dimBlock.x: %d dimGrid.x %d\n", dimBlock.x, dimGrid.x);
	cudaEventRecord(start);

	edgeDetect << <dimGrid, dimBlock >> >
		(pts1, edges1, 13568);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("run_kernel1 launch failed\n");
		printf("dimBlock: %d, %d\n", dimBlock.x, dimBlock.y);
		printf("dimGrid: %d, %d\n", dimGrid.x, dimGrid.y);
		printf("%s\n", cudaGetErrorString(error));
	}

	edgeDetect << <dimGrid, dimBlock >> >
		(pts2, edges2, 13568);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("run_kernel1 launch failed\n");
		printf("dimBlock: %d, %d\n", dimBlock.x, dimBlock.y);
		printf("dimGrid: %d, %d\n", dimGrid.x, dimGrid.y);
		printf("%s\n", cudaGetErrorString(error));
	}

	dimBlock.x = blockSize;
	dimGrid.x = ceil((float)COL / dimBlock.x);
	//dimBlock(blockSize, 1, 1); //threads per block
	//dimGrid(ceil((float)COL / dimBlock.x)); //number of blocks
	printf("dimBlock.x: %d dimGrid.x %d\n", dimBlock.x, dimGrid.x);

	//kernelNN2 << <dimGrid, dimBlock >> >(pts1, pts2, distances, closest);
	kernelNNStructured << <dimGrid, dimBlock >> > (pts1, pts2, distances, closest, 7);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("run_kernel1 launch failed\n");
		printf("dimBlock: %d, %d\n", dimBlock.x, dimBlock.y);
		printf("dimGrid: %d, %d\n", dimGrid.x, dimGrid.y);
		printf("%s\n", cudaGetErrorString(error));
	}

	//kernelRmDup << <dimGrid, dimBlock >> > (closest, distances);
	rmEdgesFromCorr << <dimGrid, dimBlock >> > (closest, edges1, edges2);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("run_kernel1 launch failed\n");
		printf("dimBlock: %d, %d\n", dimBlock.x, dimBlock.y);
		printf("dimGrid: %d, %d\n", dimGrid.x, dimGrid.y);
		printf("%s\n", cudaGetErrorString(error));
	}

	return time;
}

float run_procedure(float *valse, float *balse, int *cpu_ptrclosest) {
	float *gpu_ptr1;
	float *gpu_ptr2;
	float *gpu_ptrdist;
	int *gpu_ptrclosest;
	int *gpu_ptredges1; //COL 1 if edge, 0 if not
	int *gpu_ptredges2; //COL 1 if edge, 0 if not

	cudaError_t error = cudaMalloc(&gpu_ptr1, ROW*COL * sizeof(float));
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("1\n");
	}


	error = cudaMalloc(&gpu_ptredges1, COL * sizeof(int)); //remove
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("1\n");
	}


	error = cudaMalloc(&gpu_ptr2, ROW*COL * sizeof(float));
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("2\n");
	}

	error = cudaMalloc(&gpu_ptredges2, COL * sizeof(int)); //remove
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("1\n");
	}

	error = cudaMalloc(&gpu_ptrdist, COL * sizeof(float));
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("3\n");
	}

	error = cudaMalloc(&gpu_ptrclosest, COL * sizeof(int));
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("4\n");
	}

	error = cudaMemcpy(gpu_ptr1, valse, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("5\n");
	}

	error = cudaMemcpy(gpu_ptr2, balse, ROW*COL * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("6\n");
	}
	
	float executiontime = run_kernel(gpu_ptr1, gpu_ptr2, gpu_ptrdist, gpu_ptrclosest, gpu_ptredges1, gpu_ptredges2);

	error = cudaMemcpy(cpu_ptrclosest, gpu_ptrclosest, COL * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("7\n");
	}
	error = cudaFree(gpu_ptr1);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("8\n");
	}
	error = cudaFree(gpu_ptredges1);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("8\n");
	}
	error = cudaFree(gpu_ptr2);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("9\n");
	}
	error = cudaFree(gpu_ptredges2);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("8\n");
	}
	error = cudaFree(gpu_ptrdist);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("10\n");
	}
	error = cudaFree(gpu_ptrclosest);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
		printf("11\n");
	}
	cudaDeviceSynchronize();
	cudaDeviceReset();

	return executiontime;
}