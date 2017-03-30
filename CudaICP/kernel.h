float run_kernel(float *pts1, float *pts2, float* distances, int *closest, int *edges1, int *edges2);

float NN_GPU(float *pts1, float *pts2, int *cpu_ptrclosest);

void normals_GPU(float* model, float *normals);
