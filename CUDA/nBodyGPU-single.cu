#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define G 6.67e-11
#define dt 1e-3
#define EPS2 0
#define BLOCK_SIZE 512

using namespace std;

__global__ void calculate_forces(float4* _p, float4* _v, float4* _f, float* _m, int N)
{
	__shared__ float4 sp[BLOCK_SIZE];

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	float m = _m[index];
	float4 f = _f[index]; float4 pos = _p[index]; float4 vel = _v[index];

	float dist, mag;
	float4 dir = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	int ind = 0; int i = 0; int j = 0; int jj = 0;

	for (i = 0; i < N / BLOCK_SIZE; i++, ind += BLOCK_SIZE)
	{
		sp[threadIdx.x] = _p[ind + threadIdx.x];
		__syncthreads();
		for (jj = 0; jj < BLOCK_SIZE; jj++)
		{
			j = ind + jj;
			//if (j > index && index < N - 1)
			if (index != j)
			{
				// Вычисление силы, действующей на тело i со стороны j
				dist = sqrtf(powf(pos.x - sp[jj].x, 2) + powf(pos.y - sp[jj].y, 2) + powf(pos.z - sp[jj].z, 2)) + EPS2;
				mag = (G * m * _m[j]) / powf(dist, 2);
				dir.x = sp[jj].x - pos.x;
				dir.y = sp[jj].y - pos.y;
				dir.z = sp[jj].z - pos.z;

				// Сумма сил, действующих на тело i
				f.x += mag * dir.x / dist;
				f.y += mag * dir.y / dist;
				f.z += mag * dir.z / dist;
			}
		}
		__syncthreads();
	}
	_f[index] = f;
}

__global__ void move_particles(float4* _p, struct float4* _v, struct float4* _f, float* _m, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float4 f = _f[index]; float4 v = _v[index]; float4 p = _p[index];
	float m = _m[index];

	float4 dv;
	dv.x = f.x / m * dt;
	dv.y = f.y / m * dt;
	dv.z = f.z / m * dt;
	float4 dp;
	dp.x = (v.x + dv.x / 2) * dt;
	dp.y = (v.y + dv.y / 2) * dt;
	dp.z = (v.z + dv.z / 2) * dt;
	v.x += dv.x;
	v.y += dv.y;
	v.z += dv.z;
	p.x += dp.x;
	p.y += dp.y;
	p.z += dp.z;
	f.x = f.y = f.z = 0;

	_p[index] = p;
	_v[index] = v;
	_f[index] = f;
}

int main(int argc, char* argv[])
{
	char filename[50];
	int i, N;
	int n_mass[7] = { 256, 4096, 16384, 65536, 131072, 262144, 524288 };

	printf("Select array: \n");
	for (i = 0; i < 7; i++)
		printf("%d: n = %d\n", i + 1, n_mass[i]);
	scanf("%d", &i);
	if (i >= 1 && i <= 7)
		N = n_mass[i - 1];
	else
		return 0;

	float4* p = (float4*)malloc(sizeof(*p) * N);
	float4* v = (float4*)malloc(sizeof(*v) * N);
	float4* f = (float4*)malloc(sizeof(*f) * N);
	float* m = (float*)malloc(sizeof(*m) * N);


	sprintf(filename, "data\\in_N%d.txt", N);
	FILE* file = fopen(filename, "r");
	fscanf(file, "%d", &N); fscanf(file, "%d", &N);
	for (i = 0; i < N; i++)
	{
		fscanf(file, "%f%f%f%f%f%f%f", &p[i].x, &p[i].y, &p[i].z, &v[i].x, &v[i].y, &v[i].z, &m[i]);
		f[i].x = f[i].y = f[i].z = 0;
	}

	float4* dev_p, * dev_v, * dev_f;
	float* dev_m;
	float4* temp_p = NULL;
	float4* temp_v = NULL;

	cudaMalloc((void**)&dev_f, N * sizeof(float4));
	cudaMalloc((void**)&dev_m, N * sizeof(float));
	cudaMalloc((void**)&dev_p, N * sizeof(float4));
	cudaMalloc((void**)&dev_v, N * sizeof(float4));


	cudaMemcpy(dev_f, f, N * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p, p, N * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v, v, N * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_m, m, N * sizeof(float), cudaMemcpyHostToDevice);
	
	printf("\nNBody moving in progress (block size = %d, n = %d)...\n\n", BLOCK_SIZE, N);
	// Замер времени
	cudaEvent_t start, stop;
	float timeGPU = 0.0f;
	// Создание событий для замера времени
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int it = 0; it < 2; it++) // Цикл по времени (модельному)
	{
		calculate_forces << <dim3(N / BLOCK_SIZE), dim3(BLOCK_SIZE) >> > (dev_p, dev_v, dev_f, dev_m, N);
		move_particles << <dim3(N / BLOCK_SIZE), dim3(BLOCK_SIZE) >> > (dev_p, dev_v, dev_f, dev_m, N);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeGPU, start, stop);
	
	cudaMemcpy(p, dev_p, N * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(v, dev_v, N * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(f, dev_f, N * sizeof(float4), cudaMemcpyDeviceToHost);
	// Вывод времени на экран
	printf("# NBody (n=%d)\n", N);
	printf("# Elapsed time GPU (sec): %.3f s\n", timeGPU / 1000);

	if (filename)
	{
		sprintf(filename, "data\\outGPU-single_N%d.csv", N);
		FILE* fout = fopen(filename, "w");
		if (!fout)
		{
			fprintf(stderr, "Can't save file/n");
			exit(EXIT_FAILURE);
		}
		fprintf(fout, "x,y,z,m\n");
		for (int i = 0; i < N; i++)
		{
			fprintf(fout, "%15f,%15f,%15f,%15f\n", p[i].x, p[i].y, p[i].z, m[i]);
		}
		fclose(fout);
	}

	cudaFree(dev_m);
	cudaFree(dev_f);
	cudaFree(dev_p);
	cudaFree(dev_v);
	cudaFree(m);

	delete(f);
	delete(v);
	delete(p);
	delete(m);

	return 0;
}