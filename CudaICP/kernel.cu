#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <limits>
#include <ctime>

#include <iostream>
#include <fstream>
#include <sstream>

#include <thread>

#include <cublas_v2.h>

#include "kernel.h"

#define ROW 3 //the dimension
#define COL 217088 //number of points

#define MAX(a, b) ((a)>(b)?(a):(b))
#define n 3

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define PERR(call) \
  if (call) {\
    fprintf(stderr, "%s:%d Error [%s] on "#call"\n", __FILE__, __LINE__,\
      cudaGetErrorString(cudaGetLastError()));\
	system("pause");\
    exit(1);\
  }

#define ERRCHECK \
  if (cudaPeekAtLastError()) { \
    fprintf(stderr, "%s:%d Error [%s]\n", __FILE__, __LINE__,\
       cudaGetErrorString(cudaGetLastError()));\
	system("pause");\
    exit(1);\
  }

__device__ const int numPoints = 217088;
__device__ const int blockSize = 512;
int threads = blockSize;

//Helper-function for eigen_decompositon
__device__ void tred2(double V[n][n], double d[n], double e[n]) {

	//  This is derived from the Algol procedures tred2 by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	for (int j = 0; j < n; j++) {
		d[j] = V[n - 1][j];
	}

	// Householder reduction to tridiagonal form.

	for (int i = n - 1; i > 0; i--) {

		// Scale to avoid under/overflow.

		double scale = 0.0;
		double h = 0.0;
		for (int k = 0; k < i; k++) {
			scale = scale + fabs(d[k]);
		}
		if (scale == 0.0) {
			e[i] = d[i - 1];
			for (int j = 0; j < i; j++) {
				d[j] = V[i - 1][j];
				V[i][j] = 0.0;
				V[j][i] = 0.0;
			}
		}
		else {

			// Generate Householder vector.

			for (int k = 0; k < i; k++) {
				d[k] /= scale;
				h += d[k] * d[k];
			}
			double f = d[i - 1];
			double g = sqrt(h);
			if (f > 0) {
				g = -g;
			}
			e[i] = scale * g;
			h = h - f * g;
			d[i - 1] = f - g;
			for (int j = 0; j < i; j++) {
				e[j] = 0.0;
			}

			// Apply similarity transformation to remaining columns.

			for (int j = 0; j < i; j++) {
				f = d[j];
				V[j][i] = f;
				g = e[j] + V[j][j] * f;
				for (int k = j + 1; k <= i - 1; k++) {
					g += V[k][j] * d[k];
					e[k] += V[k][j] * f;
				}
				e[j] = g;
			}
			f = 0.0;
			for (int j = 0; j < i; j++) {
				e[j] /= h;
				f += e[j] * d[j];
			}
			double hh = f / (h + h);
			for (int j = 0; j < i; j++) {
				e[j] -= hh * d[j];
			}
			for (int j = 0; j < i; j++) {
				f = d[j];
				g = e[j];
				for (int k = j; k <= i - 1; k++) {
					V[k][j] -= (f * e[k] + g * d[k]);
				}
				d[j] = V[i - 1][j];
				V[i][j] = 0.0;
			}
		}
		d[i] = h;
	}

	// Accumulate transformations.

	for (int i = 0; i < n - 1; i++) {
		V[n - 1][i] = V[i][i];
		V[i][i] = 1.0;
		double h = d[i + 1];
		if (h != 0.0) {
			for (int k = 0; k <= i; k++) {
				d[k] = V[k][i + 1] / h;
			}
			for (int j = 0; j <= i; j++) {
				double g = 0.0;
				for (int k = 0; k <= i; k++) {
					g += V[k][i + 1] * V[k][j];
				}
				for (int k = 0; k <= i; k++) {
					V[k][j] -= g * d[k];
				}
			}
		}
		for (int k = 0; k <= i; k++) {
			V[k][i + 1] = 0.0;
		}
	}
	for (int j = 0; j < n; j++) {
		d[j] = V[n - 1][j];
		V[n - 1][j] = 0.0;
	}
	V[n - 1][n - 1] = 1.0;
	e[0] = 0.0;
}

//Helper function for eigen_decompositon
__device__ void tql2(double V[n][n], double d[n], double e[n]) {
	// Symmetric tridiagonal QL algorithm.

	//  This is derived from the Algol procedures tql2, by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	for (int i = 1; i < n; i++) {
		e[i - 1] = e[i];
	}
	e[n - 1] = 0.0;

	double f = 0.0;
	double tst1 = 0.0;
	double eps = pow(2.0, -52.0);
	for (int l = 0; l < n; l++) {

		// Find small subdiagonal element

		tst1 = MAX(tst1, fabs(d[l]) + fabs(e[l]));
		int m = l;
		while (m < n) {
			if (fabs(e[m]) <= eps*tst1) {
				break;
			}
			m++;
		}

		// If m == l, d[l] is an eigenvalue,
		// otherwise, iterate.

		if (m > l) {
			int iter = 0;
			do {
				iter = iter + 1;  // (Could check iteration count here.)

								  // Compute implicit shift

				double g = d[l];
				double p = (d[l + 1] - g) / (2.0 * e[l]);
				double r = sqrt(p*p + 1);//hypot2(p, 1.0);
				if (p < 0) {
					r = -r;
				}
				d[l] = e[l] / (p + r);
				d[l + 1] = e[l] * (p + r);
				double dl1 = d[l + 1];
				double h = g - d[l];
				for (int i = l + 2; i < n; i++) {
					d[i] -= h;
				}
				f = f + h;

				// Implicit QL transformation.

				p = d[m];
				double c = 1.0;
				double c2 = c;
				double c3 = c;
				double el1 = e[l + 1];
				double s = 0.0;
				double s2 = 0.0;
				for (int i = m - 1; i >= l; i--) {
					c3 = c2;
					c2 = c;
					s2 = s;
					g = c * e[i];
					h = c * p;
					r = sqrt(p*p + e[i] * e[i]);//hypot2(p,e[i]);
					e[i + 1] = s * r;
					s = e[i] / r;
					c = p / r;
					p = c * d[i] - s * g;
					d[i + 1] = h + s * (c * g + s * d[i]);

					// Accumulate transformation.

					for (int k = 0; k < n; k++) {
						h = V[k][i + 1];
						V[k][i + 1] = s * V[k][i] + c * h;
						V[k][i] = c * V[k][i] - s * h;
					}
				}
				p = -s * s2 * c3 * el1 * e[l] / dl1;
				e[l] = s * p;
				d[l] = c * p;

				// Check for convergence.

			} while (fabs(e[l]) > eps*tst1);
		}
		d[l] = d[l] + f;
		e[l] = 0.0;
	}

	// Sort eigenvalues and corresponding vectors.

	for (int i = 0; i < n - 1; i++) {
		int k = i;
		double p = d[i];
		for (int j = i + 1; j < n; j++) {
			if (d[j] < p) {
				k = j;
				p = d[j];
			}
		}
		if (k != i) {
			d[k] = d[i];
			d[i] = p;
			for (int j = 0; j < n; j++) {
				p = V[j][i];
				V[j][i] = V[j][k];
				V[j][k] = p;
			}
		}
	}
}

//Inverts the symmetric 3x3 matrix A, where A is row-major
__device__ void syminverse3x3(double *A) {
	double A_temp[9];

	double det = A[0] * A[4] * A[8] + 2 * A[1] * A[2] * A[5] - A[4] * A[2] * A[2] - A[0] * A[5] * A[5] - A[8] * A[1] * A[1];
	double invdet = 1 / det;

	A_temp[0] = (A[4] * A[8] - A[5] * A[5])*invdet;
	A_temp[1] = -(A[1] * A[8] - A[2] * A[5])*invdet;
	A_temp[2] = (A[1] * A[5] - A[2] * A[4])*invdet;
	A_temp[3] = A_temp[1];
	A_temp[4] = (A[0] * A[8] - A[2] * A[2])*invdet;
	A_temp[5] = -(A[0] * A[5] - A[1] * A[2])*invdet;
	A_temp[6] = A_temp[2];
	A_temp[7] = A_temp[5];
	A_temp[8] = (A[0] * A[4] - A[1] * A[1])*invdet;


	for (int i = 0; i < 9; i++) {
		A[i] = A_temp[i];
	}
}

//Calculates transformation matrix from 6dof parameters (roll,pitch,yaw,tx,ty,tz)
__host__ __device__ void pose_from6dof(float *par, float *T) {

	T[0] = cos(par[2])*cos(par[1]);
	T[1] = -sin(par[2])*cos(par[0]) + cos(par[2])*sin(par[1])*sin(par[0]);
	T[2] = sin(par[2])*sin(par[0]) + cos(par[2])*sin(par[1])*cos(par[0]);
	T[3] = par[3];
	T[4] = sin(par[2])*cos(par[1]);
	T[5] = cos(par[2])*cos(par[0]) + sin(par[2])*sin(par[1])*sin(par[0]);
	T[6] = -cos(par[2])*sin(par[0]) + sin(par[2])*sin(par[1])*cos(par[0]);
	T[7] = par[4];
	T[8] = -sin(par[1]);
	T[9] = cos(par[1])*sin(par[0]);
	T[10] = cos(par[1])*cos(par[0]);
	T[11] = par[5];
}

//Finds the egenvalues d and corresponding eigenvectors V for the 3x3 symmetric matrix A
__device__ void eigen_decomposition(double A[n][n], double V[n][n], double d[n]) {
	double e[n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			V[i][j] = A[i][j];
		}
	}
	tred2(V, d, e);
	tql2(V, d, e);
}

//This Kernel finds NN between pts1 and pts2 by exploiting the structure of the point cloud
__global__ void kernelNNStructured
(float *model, float *target, float *distances, int *closest, int area) {
	// This implementation exploits the matrix structure of a depth image from a 3D camera and perfroms a limited NN search for each point. 
	// For a query point q with row coordinate a and column coordinate b in matrix pts1, this algorithm searches through the points in the range a-n:a+n b-m:b+n in matrix pts2
	// This assumes small increments in the movement between the two consecutive depth images pts1 and pts2
	// This implementation is hard-coded for a 512x424 depth image

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < numPoints) {
		float distToClosest = 999999;
		int closestIndex = -1;

		int a = idx / 512; //this row
		int b = idx % 512; //this col
		for (int i = -area; i <= area; i++) {
			for (int j = -area; j <= area; j++) {
				if (a + i >= 0 && a + i < 424 && b + j >= 0 && b + j < 512 && (i*i + j*j) <= (area + 1)*(area + 1)) { //this check makes sure we are within the limits of the 512x424 matrix and that we do a radius search instead of a square search
					int this_ind = (a + i) * 512 + b + j;
					float dist = (model[idx * 3 + 0] - target[this_ind * 3 + 0])*(model[idx * 3 + 0] - target[this_ind * 3 + 0]) +
						(model[idx * 3 + 1] - target[this_ind * 3 + 1])*(model[idx * 3 + 1] - target[this_ind * 3 + 1]) +
						(model[idx * 3 + 2] - target[this_ind * 3 + 2])*(model[idx * 3 + 2] - target[this_ind * 3 + 2]);
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

//This kernel finds the NN brute force by using the "blocking" teqnuiqe for efficient execution
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

//This kernel finds the NN brute force
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

//This kernel detect mesh boundaries by exploiting the structure of the point cloud
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
				float ind = i * 4 + j + 1;
				float this_point = pts[(pos + j + i * 512) * 3 + 2];
				if (ind > 1) {
					this_var = ((ind - 2) / (ind - 1))*this_var + (1 / ind)*(this_point - prev_mean)*(this_point - prev_mean);
				}
				prev_mean = (this_point + (ind - 1)*prev_mean) / ind;
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

//This kernel removes correspondences on the mesh boundary from the correspondences list
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

//This kernel removes duplicate correspondences from the correspondences and keeps the correspondences with the smallest distance
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

//Computes the approx normals for by using Ax=b and setting all values of b equal to the distance from origin to the point
__global__ void
approxNormals(float *pts, float *norms, int radius)
{
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < numPoints) {
		if (isfinite(pts[idx * 3])) {
			int a = idx / 512; //this row
			int b = idx % 512; //this col
			double AtA[9] = { 0,0,0 , 0,0,0, 0,0,0 }; //rowmajor 3x3 matrix
			double Atd[3] = { 0,0,0 };
			float d = sqrt(pts[idx * 3 + 0] * pts[idx * 3 + 0] + pts[idx * 3 + 1] * pts[idx * 3 + 1] + pts[idx * 3 + 2] + pts[idx * 3 + 2]);
			for (int i = -radius; i <= radius; i++) {
				for (int j = -radius; j <= radius; j++) {
					if (a + i >= 0 && a + i < 424 && b + j >= 0 && b + j < 512 && (i*i + j*j) <= (radius + 1)*(radius + 1)) { //this check makes sure we are within the limits of the 512x424 matrix and that we do a radius search instead of a square search
						int this_ind = (a + i) * 512 + b + j;
						if (isfinite(pts[this_ind * 3])) {
							AtA[0] += pts[this_ind * 3 + 0] * pts[this_ind * 3 + 0];
							AtA[4] += pts[this_ind * 3 + 1] * pts[this_ind * 3 + 1];
							AtA[8] += pts[this_ind * 3 + 2] * pts[this_ind * 3 + 2];
							AtA[1] += pts[this_ind * 3 + 0] * pts[this_ind * 3 + 1];
							AtA[2] += pts[this_ind * 3 + 0] * pts[this_ind * 3 + 2];
							AtA[5] += pts[this_ind * 3 + 1] * pts[this_ind * 3 + 2];

							Atd[0] += d*pts[this_ind * 3 + 0];
							Atd[1] += d*pts[this_ind * 3 + 1];
							Atd[2] += d*pts[this_ind * 3 + 2];
						}
					}
				}
			}
			AtA[3] = AtA[1];
			AtA[6] = AtA[2];
			AtA[7] = AtA[5];

			syminverse3x3(AtA);

			float norm = 0;
			for (int i = 0; i < 3; i++) {
				norms[idx * 3 + i] = AtA[i * 3 + 0] * Atd[0] + AtA[i * 3 + 1] * Atd[1] + AtA[i * 3 + 2] * Atd[2];
				norm += norms[idx * 3 + i] * norms[idx * 3 + i];
			}
			//normalize to create unit normal
			norm = sqrt(norm);
			for (int i = 0; i < 3; i++) {
				norms[idx * 3 + i] = norms[idx * 3 + i] / norm;
			}
			//direct normal towards camera viewpoint
			if ((-norms[idx * 3 + 0] * pts[idx * 3 + 0] - norms[idx * 3 + 1] * pts[idx * 3 + 1] - norms[idx * 3 + 2] * pts[idx * 3 + 2]) >= 0) {
				norms[idx * 3 + 0] = -norms[idx * 3 + 0];
				norms[idx * 3 + 1] = -norms[idx * 3 + 1];
				norms[idx * 3 + 2] = -norms[idx * 3 + 2];
			}
		}
		else {
			for (int i = 0; i < 3; i++) {
				norms[idx * 3 + i] = 0;
			}
		}

	}
}

//Computes the exact normals by using the eigenvectors for the covariance-matrix
__global__ void
exactNormals(float *pts, float *norms, int radius)
{
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < numPoints) {
		if (isfinite(pts[idx * 3])) {
			int a = idx / 512; //this row
			int b = idx % 512; //this col
			double M[n][n] = { {0,0,0},{ 0,0,0 },{ 0,0,0 } };
			double V[n][n];
			double d[n];
			double mean[n] = { 0,0,0 };
			int num = 0;
			//First calculate mean
			for (int i = -radius; i <= radius; i++) {
				for (int j = -radius; j <= radius; j++) {
					if (a + i >= 0 && a + i < 424 && b + j >= 0 && b + j < 512 && (i*i + j*j) <= (radius + 1)*(radius + 1)) { //this check makes sure we are within the limits of the 512x424 matrix and that we do a radius search instead of a square search
						int this_ind = (a + i) * 512 + b + j;
						if (isfinite(pts[this_ind * 3])) {
							mean[0] += pts[this_ind * 3 + 0]; mean[1] += pts[this_ind * 3 + 1]; mean[2] += pts[this_ind * 3 + 2];
							num++;
						}
					}
				}
			}
			mean[0] /= num; mean[1] /= num; mean[2] /= num;
			//Then calculate covariance matrix
			for (int i = -radius; i <= radius; i++) {
				for (int j = -radius; j <= radius; j++) {
					if (a + i >= 0 && a + i < 424 && b + j >= 0 && b + j < 512 && (i*i + j*j) <= (radius + 1)*(radius + 1)) { //this check makes sure we are within the limits of the 512x424 matrix and that we do a radius search instead of a square search
						int this_ind = (a + i) * 512 + b + j;
						if (isfinite(pts[this_ind * 3])) {
							float a = pts[this_ind * 3 + 0] - mean[0];
							float b = pts[this_ind * 3 + 1] - mean[1];
							float c = pts[this_ind * 3 + 2] - mean[2];
							M[0][0] += a*a; M[0][1] += a*b; M[0][2] += a*c;
							M[1][0] = M[0][1]; M[1][1] += b*b; M[1][2] += b*c;
							M[2][0] = M[0][2];  M[2][1] = M[1][2];  M[2][2] += c*c;
						}
					}
				}
			}
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					M[i][j] /= num;
				}
			}
			//Find egenvalues and eigenvectors
			eigen_decomposition(M, V, d);
			//The normal is equal to the smallest eigenvalue
			for (int i = 0; i < 3; i++) {
				norms[idx * 3 + i] = V[i][0];
			}
			//direct normal towards camera viewpoint
			if ((-norms[idx * 3 + 0] * pts[idx * 3 + 0] - norms[idx * 3 + 1] * pts[idx * 3 + 1] - norms[idx * 3 + 2] * pts[idx * 3 + 2]) >= 0) {
				norms[idx * 3 + 0] = -norms[idx * 3 + 0];
				norms[idx * 3 + 1] = -norms[idx * 3 + 1];
				norms[idx * 3 + 2] = -norms[idx * 3 + 2];
			}
		}
		else {
			for (int i = 0; i < 3; i++) {
				norms[idx * 3 + i] = 0;
			}
		}
	}
}

//This kernerl sets up the Point to Plane ICP problem on the form Ax=b
__global__ void setupPointToPlane(float *model, float *target, float *normals, int *closest, float *A, float *b) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < numPoints) {
		if (closest[idx] != -1) {
			b[idx] = normals[closest[idx] * 3 + 0] * target[closest[idx] * 3 + 0] + normals[closest[idx] * 3 + 1] * target[closest[idx] * 3 + 1] + normals[closest[idx] * 3 + 2] * target[closest[idx] * 3 + 2]
				- normals[closest[idx] * 3 + 0] * model[idx * 3 + 0] - normals[closest[idx] * 3 + 1] * model[idx * 3 + 1] - normals[closest[idx] * 3 + 2] * model[idx * 3 + 2];

			//row major
			/*A[idx * 6 + 0] = normals[closest[idx] * 3 + 2] * model[idx * 3 + 1] - normals[closest[idx] * 3 + 1] * model[idx * 3 + 2];
			A[idx * 6 + 1] = normals[closest[idx] * 3 + 0] * model[idx * 3 + 2] - normals[closest[idx] * 3 + 2] * model[idx * 3 + 0];
			A[idx * 6 + 2] = normals[closest[idx] * 3 + 1] * model[idx * 3 + 0] - normals[closest[idx] * 3 + 0] * model[idx * 3 + 1];
			A[idx * 6 + 3] = normals[closest[idx] * 3 + 0];
			A[idx * 6 + 4] = normals[closest[idx] * 3 + 1];
			A[idx * 6 + 5] = normals[closest[idx] * 3 + 2];*/

			//column major for cublas library
			A[IDX2C(idx, 0, COL)] = normals[closest[idx] * 3 + 2] * model[idx * 3 + 1] - normals[closest[idx] * 3 + 1] * model[idx * 3 + 2];
			A[IDX2C(idx, 1, COL)] = normals[closest[idx] * 3 + 0] * model[idx * 3 + 2] - normals[closest[idx] * 3 + 2] * model[idx * 3 + 0];
			A[IDX2C(idx, 2, COL)] = normals[closest[idx] * 3 + 1] * model[idx * 3 + 0] - normals[closest[idx] * 3 + 0] * model[idx * 3 + 1];
			A[IDX2C(idx, 3, COL)] = normals[closest[idx] * 3 + 0];
			A[IDX2C(idx, 4, COL)] = normals[closest[idx] * 3 + 1];
			A[IDX2C(idx, 5, COL)] = normals[closest[idx] * 3 + 2];
		}
		else {
			b[idx] = 0;
			/*A[idx * 6 + 0] = 0;
			A[idx * 6 + 1] = 0;
			A[idx * 6 + 2] = 0;
			A[idx * 6 + 3] = 0;
			A[idx * 6 + 4] = 0;
			A[idx * 6 + 5] = 0;*/
			A[IDX2C(idx, 0, COL)] = 0;
			A[IDX2C(idx, 1, COL)] = 0;
			A[IDX2C(idx, 2, COL)] = 0;
			A[IDX2C(idx, 3, COL)] = 0;
			A[IDX2C(idx, 4, COL)] = 0;
			A[IDX2C(idx, 5, COL)] = 0;
		}
	}
}

//This kernel transform the point cloud "pts" using 6dof par [roll,pitch,yaw,t1,t2,t3]
__global__ void transformCloud(float *pts, float *par) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < numPoints) {
		if (isfinite(pts[idx * 3])) {
			float T[12];
			pose_from6dof(par, T);
			float temp[3] = { pts[idx * 3 + 0],pts[idx * 3 + 1],pts[idx * 3 + 2] };
			pts[idx * 3 + 0] = T[0] * temp[0] + T[1] * temp[1] + T[2] * temp[2] + T[3];
			pts[idx * 3 + 1] = T[4] * temp[0] + T[5] * temp[1] + T[6] * temp[2] + T[7];
			pts[idx * 3 + 2] = T[8] * temp[0] + T[9] * temp[1] + T[10] * temp[2] + T[11];
		}
	}
}

void getDataFromFile(float *data, std::string name) {
	std::fstream in;
	in.open(name);
	if (in.fail()) {
		printf("error loading .txt file reading\n");
		return;
	}
	std::string line;
	std::string value;
	int i = 0;
	while (std::getline(in, line)) {
		std::stringstream ss(line);
		while (ss >> value) {
			if (value == "-inf") {
				data[i] = -std::numeric_limits<float>::infinity();
			}
			else {
				data[i] = std::stof(value);
			}
			i++;
		}
	}
	in.close();
}

void updateAccumulatedTransformation(float *par, float *T) {
	float ti[12];
	pose_from6dof(par, ti);
	float temp[12];
	for (int i = 0; i < 12; i++) {
		temp[i] = T[i];
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			T[i * 4 + j] = ti[i * 4] * temp[j] + ti[i * 4 + 1] * temp[4+j] + ti[i * 4 + 2] * temp[8+j] + (float)(((int)j / 3)*ti[i * 4 + 3]);
		}
	}
}

//Computes the normals on the GPU by exploiting the structure of the depth image
void normals_GPU(float *model, float *normals) {
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float *gpu_model, *gpu_normals;
	dim3 dimBlock(512, 1, 1);
	dim3 dimGrid(ceil((float)COL / dimBlock.x), 1, 1);
	PERR(cudaMalloc(&gpu_model, ROW*COL * sizeof(float)));
	PERR(cudaMalloc(&gpu_normals, ROW*COL * sizeof(float)));
	PERR(cudaMemcpy(gpu_model, model, ROW*COL * sizeof(float), cudaMemcpyHostToDevice));
	cudaEventRecord(start); //start timer
	//approxNormals << <dimGrid, dimBlock >> > (gpu_model, gpu_normals, 5);
	exactNormals << <dimGrid, dimBlock >> > (gpu_model, gpu_normals, 5);
	cudaEventRecord(stop); //stop timer
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	ERRCHECK;
	PERR(cudaMemcpy(normals, gpu_normals, ROW*COL * sizeof(float), cudaMemcpyDeviceToHost));
	PERR(cudaFree(gpu_model));
	PERR(cudaFree(gpu_normals));
	cudaDeviceSynchronize();
	printf("norm time: %f\n", time);
}

//Runs kernels after memory is allocated
float run_kernel(float *d_model, float *d_target, float* d_distances, int *d_closest, int *d_modelEdges, int *d_targetEdges)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 dimBlock(16, 1, 1); //threads per block
	dim3 dimGrid(ceil((float)COL / dimBlock.x)); //number of blocks
	printf("dimBlock.x: %d dimGrid.x %d\n", dimBlock.x, dimGrid.x);

	cudaEventRecord(start); //start timer

	edgeDetect << <dimGrid, dimBlock >> > (d_model, d_modelEdges, 13568);
	ERRCHECK;
	edgeDetect << <dimGrid, dimBlock >> > (d_target, d_targetEdges, 13568);
	ERRCHECK;

	dimBlock.x = blockSize;
	dimGrid.x = ceil((float)COL / dimBlock.x);
	printf("dimBlock.x: %d dimGrid.x %d\n", dimBlock.x, dimGrid.x);

	//kernelNN2 << <dimGrid, dimBlock >> >(pts1, pts2, distances, closest);
	kernelNNStructured << <dimGrid, dimBlock >> > (d_model, d_target, d_distances, d_closest, 7);
	ERRCHECK;
	//kernelRmDup << <dimGrid, dimBlock >> > (closest, distances);
	rmEdgesFromCorr << <dimGrid, dimBlock >> > (d_closest, d_modelEdges, d_targetEdges);
	ERRCHECK;
	cudaEventRecord(stop); //stop timer
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	return time;
}

//This function is called from main and runs the entire procedure of NN detection, rejection and normal computation
float NN_GPU(float *model, float *target, int *closest) {
	float *gpuptr_model;
	float *gpuptr_target;
	float *gpuptr_dist;
	int *gpuptr_closest;
	int *gpuptr_modelEdges; //COL 1 if edge, 0 if not
	int *gpuptr_targetEdges; //COL 1 if edge, 0 if not

	PERR(cudaMalloc(&gpuptr_model, ROW*COL * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_modelEdges, COL * sizeof(int)));
	PERR(cudaMalloc(&gpuptr_target, ROW*COL * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_targetEdges, COL * sizeof(int)));
	PERR(cudaMalloc(&gpuptr_dist, COL * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_closest, COL * sizeof(int)));

	PERR(cudaMemcpy(gpuptr_model, model, ROW*COL * sizeof(float), cudaMemcpyHostToDevice));
	PERR(cudaMemcpy(gpuptr_target, target, ROW*COL * sizeof(float), cudaMemcpyHostToDevice));

	float executiontime = run_kernel(gpuptr_model, gpuptr_target, gpuptr_dist, gpuptr_closest, gpuptr_modelEdges, gpuptr_targetEdges);

	PERR(cudaMemcpy(closest, gpuptr_closest, COL * sizeof(int), cudaMemcpyDeviceToHost));

	PERR(cudaFree(gpuptr_model));
	PERR(cudaFree(gpuptr_modelEdges));
	PERR(cudaFree(gpuptr_target));
	PERR(cudaFree(gpuptr_targetEdges));
	PERR(cudaFree(gpuptr_dist));
	PERR(cudaFree(gpuptr_closest));

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return executiontime;
}

float PTP_GPU() {
	float *gpuptr_model, *gpuptr_target, *gpuptr_dist, *gpuptr_normals, *gpuptr_A, *gpuptr_AtA, *gpuptr_b, *gpuptr_Atb,*gpuptr_T;
	int *gpuptr_closest;
	int *gpuptr_modelEdges; //COL 1 if edge, 0 if not
	int *gpuptr_targetEdges; //COL 1 if edge, 0 if not

	float **aAtA;
	int* dLUPivots;
	int* dLUInfo;
	float *a, *be;
	a = (float*)malloc(sizeof(gpuptr_model));
	be = (float*)malloc(sizeof(gpuptr_model));
	*a = 1.0;
	*be = 0.0;

	float *h_par = new float[6];
	float *target = new float[ROW*COL];
	int icp_iterations = 20;
	int datasets = 100;
	std::clock_t startTime;
	std::clock_t endTime;

	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate_v2(&handle);
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 dimBlock(16, 1, 1); //threads per block
	dim3 dimGrid(ceil((float)COL / dimBlock.x)); //number of blocks

	PERR(cudaMalloc(&aAtA, sizeof(float*)));
	PERR(cudaMalloc(&dLUPivots, 6 * sizeof(int)));
	PERR(cudaMalloc(&dLUInfo, sizeof(int)));

	PERR(cudaMalloc(&gpuptr_model, ROW*COL * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_modelEdges, COL * sizeof(int)));
	PERR(cudaMalloc(&gpuptr_target, ROW*COL * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_targetEdges, COL * sizeof(int)));
	PERR(cudaMalloc(&gpuptr_dist, COL * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_normals, ROW*COL * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_closest, COL * sizeof(int)));
	PERR(cudaMalloc(&gpuptr_A, 2*ROW*COL * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_T, 11 * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_AtA, 36 * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_Atb, 6 * sizeof(float)));
	PERR(cudaMalloc(&gpuptr_b, COL * sizeof(float)));

	std::string name = "Cloud_uncompressed0.txt";
	std::thread t1(getDataFromFile, target, name);
	t1.join();
	PERR(cudaMemcpy(gpuptr_model, target, ROW*COL * sizeof(float), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	name = "Cloud_uncompressed1.txt";
	std::thread t2(getDataFromFile, target, name);
	t2.join();
	PERR(cudaMemcpy(gpuptr_target, target, ROW*COL * sizeof(float), cudaMemcpyHostToDevice));


	//Loop over data-sets here
	///////////////////////////////////////////////////////////
	cudaEventRecord(start); //start timer
	for (int cld_num = 2; cld_num < datasets; cld_num++) {
		//get new target for next iteration
		name = "Cloud_uncompressed" + std::to_string(cld_num) + ".txt";
		std::thread t3(getDataFromFile, target, name);

		//These operations are only done once for the data-set
		/////////////////////////////////////////////
		edgeDetect << <dimGrid, dimBlock >> > (gpuptr_model, gpuptr_modelEdges, 13568);
		ERRCHECK;
		edgeDetect << <dimGrid, dimBlock >> > (gpuptr_target, gpuptr_targetEdges, 13568);
		ERRCHECK;

		dimBlock.x = blockSize;
		dimGrid.x = ceil((float)COL / dimBlock.x);

		exactNormals << <dimGrid, dimBlock >> > (gpuptr_target, gpuptr_normals, 5);
		ERRCHECK;
		float T[12] = { 1,0,0,0,0,1,0,0 ,0,0,1,0 };
		//Loop ICP here untill convergence
		/////////////////////////////////////////////
		for (int it = 0; it < icp_iterations; it++)
		{
			kernelNNStructured << <dimGrid, dimBlock >> > (gpuptr_model, gpuptr_target, gpuptr_dist, gpuptr_closest, 7);
			ERRCHECK;
			rmEdgesFromCorr << <dimGrid, dimBlock >> > (gpuptr_closest, gpuptr_modelEdges, gpuptr_targetEdges);
			ERRCHECK;
			setupPointToPlane << <dimGrid, dimBlock >> > (gpuptr_model, gpuptr_target, gpuptr_normals, gpuptr_closest, gpuptr_A, gpuptr_b);
			ERRCHECK;
			//Find A^T*A
			status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 6, 6, COL, a, gpuptr_A, COL, gpuptr_A, COL, be, gpuptr_AtA, 6);
			//And A^T*b
			status = cublasSgemv(handle, CUBLAS_OP_T, COL, 6, a, gpuptr_A, COL, gpuptr_b, 1, be, gpuptr_Atb, 1);
			cudaDeviceSynchronize();
			//Then invert to find (A^T*A)^-1
			PERR(cudaMemcpy(aAtA, &gpuptr_AtA, sizeof(float*), cudaMemcpyHostToDevice));
			status = cublasSgetrfBatched(handle, 6, aAtA, 6, dLUPivots, dLUInfo, 1);
			status = cublasSgetriBatched(handle, 6, (const float **)aAtA, 6, dLUPivots, aAtA, 6, dLUInfo, 1);
			cudaDeviceSynchronize();
			//Now gpuptr_AtA is inverted!!
			//Finally find the solution (A^t*A)^-1*A^T*b (the solution is stored in gpuptr_Atb)
			status = cublasSgemv(handle, CUBLAS_OP_N, 6, 6, a, gpuptr_AtA, 6, gpuptr_Atb, 1, be, gpuptr_Atb, 1);
			cudaDeviceSynchronize();
			//Transform the model cloud according to point-to-plane solution
			transformCloud << <dimGrid, dimBlock >> > (gpuptr_model, gpuptr_Atb);
			cudaDeviceSynchronize();
			ERRCHECK;
			PERR(cudaMemcpy(h_par, gpuptr_Atb, 6 * sizeof(float), cudaMemcpyDeviceToHost));
			updateAccumulatedTransformation(h_par, T);
		}
		/////////////////////////////////////////////
		//Model becomes target
		float *temp = gpuptr_target;
		gpuptr_target = gpuptr_model;
		gpuptr_model = temp;
		//update target on gpu
		//startTime = std::clock();
		t3.join();
		//endTime = std::clock();
		//std::cout << "Wating for file to load took: " << (endTime - startTime) / (double)CLOCKS_PER_SEC << std::endl;
		PERR(cudaMemcpy(gpuptr_target, target, ROW*COL * sizeof(float), cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	}
	///////////////////////////////////////////////////////////

	cudaEventRecord(stop); //stop timer
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cublasDestroy_v2(handle);
	free(a);
	free(be);
	delete[] h_par; h_par = nullptr;
	delete[] target; target = nullptr;
	PERR(cudaFree(gpuptr_T));
	PERR(cudaFree(gpuptr_Atb));
	PERR(cudaFree(gpuptr_AtA));
	PERR(cudaFree(gpuptr_A));
	PERR(cudaFree(gpuptr_b));
	PERR(cudaFree(gpuptr_normals));
	PERR(cudaFree(gpuptr_model));
	PERR(cudaFree(gpuptr_modelEdges));
	PERR(cudaFree(gpuptr_target));
	PERR(cudaFree(gpuptr_targetEdges));
	PERR(cudaFree(gpuptr_dist));
	PERR(cudaFree(gpuptr_closest));

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return time/(datasets-2);
}