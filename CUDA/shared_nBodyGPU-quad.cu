#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <math.h>
#include <omp.h>

#define BLOCK_SIZE 512
#define G 6.67e-11
#define dt 1e-3
#define EPS2 0

using namespace std;

__global__ void move_NBody(float4* _p1, float4* _p2, float4* _p3, float4* _p4, float4* _v, float4* _f, float* _m, int N)
{
	// Массив в разделяемой памяти для хранения положений тел по индексу j
	__shared__ float4 sp[BLOCK_SIZE];
	__shared__ float4 fj[BLOCK_SIZE];
	// Индекс для каждой нити (тел по индексу i)
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// Присвоение значений массивов нитям
	float m = _m[index];
	float4 f = _f[index]; float4 p = _p1[index]; float4 v = _v[index];
	// Вспомогательные переменные
	float dist, mag;
	float4 dir = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	int ind = 0; int i = 0; int j = 0; int jj = 0; int k = 0;
	// Массив значений из памяти каждой из GPU
	float4* pp[4];
	pp[0] = _p1; pp[1] = _p2; pp[2] = _p3; pp[3] = _p4;
	fj[threadIdx.x] = { 0.0f, 0.0f, 0.0f, 0.0f };
	// Для j тел из памяти каждой из 4 GPU
	for (k = 0; k < 4; k++)
	{
		// Разбиение всех значений j из памяти GPU k по блокам
		ind = 0;
		for (i = 0; i < N / 4 / BLOCK_SIZE; i++, ind += BLOCK_SIZE)
		{
			sp[threadIdx.x] = pp[k][ind + threadIdx.x];
			__syncthreads();
			for (jj = 0; jj < BLOCK_SIZE; jj++)
			{
				// j индекс, учитывающих блок и номер элемента в блоке
				j = ind + jj;
				if (j > index/* && (k != 3 || index < (N / 4) - 1)*/)
				{
					// Вычисление силы, действующей на тело i со стороны j
					dist = sqrtf(powf(p.x - sp[jj].x, 2) + powf(p.y - sp[jj].y, 2) + powf(p.z - sp[jj].z, 2)) + EPS2;
					mag = (G * m * _m[j]) / powf(dist, 2);
					dir.x = sp[jj].x - p.x;
					dir.y = sp[jj].y - p.y;
					dir.z = sp[jj].z - p.z;
					// Сумма сил, действующих на тело i
					f.x += mag * dir.x / dist;
					f.y += mag * dir.y / dist;
					f.z += mag * dir.z / dist;
					fj[jj].x -= mag * dir.x / dist;
					fj[jj].y -= mag * dir.y / dist;
					fj[jj].z -= mag * dir.z / dist;
				}
				__syncthreads();
			}
		}
	}
	f.x += fj[threadIdx.x].x;
	f.y += fj[threadIdx.x].y;
	f.z += fj[threadIdx.x].z;
	// Изменение скорости
	float4 dv;
	dv.x = f.x / m * dt;
	dv.y = f.y / m * dt;
	dv.z = f.z / m * dt;
	// Изменение положения
	float4 dp;
	dp.x = (v.x + dv.x / 2) * dt;
	dp.y = (v.y + dv.y / 2) * dt;
	dp.z = (v.z + dv.z / 2) * dt;
	// Приращение скорости
	v.x += dv.x;
	v.y += dv.y;
	v.z += dv.z;
	// Сдвиг тела
	p.x += dp.x;
	p.y += dp.y;
	p.z += dp.z;
	f.x = f.y = f.z = 0;
	// Присвоение полученных значений массивам в глобальной памяти
	_p1[index] = p;
	_v[index] = v;
	_f[index] = f;
}

int main(int argc, char* argv[])
{
	// Название файла входного или выходного
	char filename[50];
	int n_mass[7] = { 256, 4096, 16384, 65536, 131072, 262144, 524288 };
	// Вспомогательные переменные
	int nthr, i, j, it, i2, i3, i4, N;
	// Инициализация N
	printf("Select array: \n");
	for (i = 0; i < 7; i++)
		printf("%d: n = %d\n", i + 1, n_mass[i]);
	scanf("%d", &i);
	if (i >= 1 && i <= 7)
		N = n_mass[i - 1];
	else
		return 0;
	// Массивы CPU
	float4* p = (float4*)malloc(sizeof(*p) * N);
	float4* v = (float4*)malloc(sizeof(*v) * N);
	float4* f = (float4*)malloc(sizeof(*f) * N);
	float* m = (float*)malloc(sizeof(*m) * N);
	// Массивы GPU
	float4* dev_p[4] = { NULL, NULL, NULL, NULL }; // Положение (видео) oldPos
	float4* dev_v[4] = { NULL, NULL, NULL, NULL }; // Скорость (видео) oldVel
	float4* dev_f;
	float* dev_m;
	// Информация о GPU
	cudaDeviceProp prop;
	// Количество потоков CPU
	#pragma omp parallel
	{
		nthr = omp_get_num_threads(); 
	}
	printf("----------------\n");
	printf("CPU Threads: %d\n", nthr);
	
	int deviceCount, deviceId[4];
	// Количество доступных GPU
	cudaGetDeviceCount(&deviceCount);
	printf("Device count: %d\n", deviceCount);
	printf("----------------\n\n");
	for (int i = 0; i < deviceCount; i++) 
	{
		cudaGetDeviceProperties(&prop, i);
		printf("| %d - %s |\n", i, prop.name);
	}
	// Ввод id используемых устройств
	printf("\nInput device id...\n", deviceCount - 1);
	for (i = 0; i < 4; i++) 
	{
		printf("Input device id: ");
		scanf("%d", &deviceId[i]);
		if (deviceId[i] > deviceCount - 1) 
		{ 
			printf("\nThere is no device with this ID"); 
			return 0;
		}
	}
	// Вывод всех id используемых устройств
	for (i = 0; i < 4; i++) 
		printf("deviceId[%d] = %d\n", i, deviceId[i]);
	// Могут ли получить доступ к памяти
	int can_access_peer;
	printf("\n-------------------------------------\n");
	#pragma omp parallel for schedule(static,1) private(i,j)
	for (i = 0; i < 4; i++) 
	{
		cudaSetDevice(deviceId[i]);
		for (j = 0; j < 4; j++) 
		{
			if (j != i) 
			{
				cudaDeviceCanAccessPeer(&can_access_peer, deviceId[i], deviceId[j]);
				printf("Device can access peer: %d to %d = %d\n", deviceId[i], deviceId[j], can_access_peer);
			}
		}
	}
	printf("-------------------------------------\n\n");
	// Выбор файла и заполнение массивов из него
	sprintf(filename, "data\\in_N%d.txt", N);
	FILE* file = fopen(filename, "r");
	fscanf(file, "%d", &N); fscanf(file, "%d", &N);
	for (i = 0; i < N; i++)
	{
		fscanf(file, "%f%f%f%f%f%f%f", &p[i].x, &p[i].y, &p[i].z, &v[i].x, &v[i].y, &v[i].z, &m[i]);
		f[i].x = f[i].y = f[i].z = 0;
	}
	// Выделение памяти для массивов GPUs
	#pragma omp parallel for schedule(static,1) private(i)
	for (i = 0; i < 4; i++)
	{
		// Использование видеокарты i
		cudaSetDevice(deviceId[i]);
		// p и v
		cudaMalloc((void**)&dev_p[i], N / 4 * sizeof(float4));
		cudaMalloc((void**)&dev_v[i], N / 4 * sizeof(float4));
		// Копирование i четверти массива p на видеокрту i
		cudaMemcpy(dev_p[i], p + i * N / 4, N / 4 * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_v[i], v + i * N / 4, N / 4 * sizeof(float4), cudaMemcpyHostToDevice);
	}
	// Выделение памяти под массивы f и m
	cudaMalloc((void**)&dev_f, N * sizeof(float4));
	cudaMalloc((void**)&dev_m, N * sizeof(float));
	// Копирование значченией f и m на GPU
	cudaMemcpy(dev_f, f, N * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_m, m, N * sizeof(float), cudaMemcpyHostToDevice);
	// Видеокарта 1
	cudaSetDevice(deviceId[1]);
	cudaEvent_t start, stop;
	float timeGPU = 0.0f;
	printf("\nNBody moving in progress (block size = %d, n = %d)...\n\n", BLOCK_SIZE, N);
	// Создание событий для замера времени
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	// Основной расчет
	for (int it = 0; it < 2; it++)
	{
	#pragma omp parallel for schedule(static,1) private(i,j,i2,i3,i4)
		for (i = 0; i < 4; i++) 
		{
			cudaSetDevice(deviceId[i]);
			// Прямой доступ к выделенной памяти остальных GPU
			for (j = 0; j < 4; j++) 
			{
				if (j != i) cudaDeviceEnablePeerAccess(deviceId[j], 0);
			}
			// Индексация частей массивов old с других GPU
			if (i == 0) { i2 = 1; i3 = 2; i4 = 3; }
			else if (i == 1) { i2 = 0; i3 = 3; i4 = 2; }
			else if (i == 2) { i2 = 3; i3 = 1; i4 = 0; }
			else if (i == 3) { i2 = 2; i3 = 0; i4 = 1; }
			// Перемещение частиц
			move_NBody << <dim3(N / 4 / BLOCK_SIZE), dim3(BLOCK_SIZE) >> > (dev_p[i], dev_p[i2], dev_p[i3], dev_p[i4], dev_v[i], dev_f, dev_m, N);
		}
	}
	#pragma omp parallel for schedule(static,1) private(i)
	for (i = 0; i < 4; i++) 
	{
		cudaSetDevice(deviceId[i]);
		// Копирование расчитанных значений со всех GPU на CPU
		cudaMemcpy(p + i * N / 4, dev_p[i], N / 4 * sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemcpy(v + i * N / 4, dev_v[i], N / 4 * sizeof(float4), cudaMemcpyDeviceToHost);
	}
	cudaSetDevice(deviceId[1]);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeGPU, start, stop);
	// Вывод времени на экран
	printf("# NBody (n=%d)\n", N);
	printf("# Elapsed time GPU (sec): %.3f s\n", timeGPU / 1000);
	//Запись полученных результатов в файл
	if (filename)
	{
		sprintf(filename, "data\\outGPU-quad_N%d.csv", N);
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
	// Очистка памяти CPU
	delete(f);
	delete(v);
	delete(p);
	delete(m);
	// Отключить доступ к выделенной памяти остальных GPU
	#pragma omp parallel for schedule(static,1) private(i)
	for (i = 0; i < 4; i++)
	{
		cudaSetDevice(deviceId[i]);
		for (j = 0; j < 4; j++)
		{
			if (j != i) cudaDeviceDisablePeerAccess(deviceId[j]);
		}
	}
	// Очистка памяти GPU
	cudaFree(dev_m);
	cudaFree(dev_f);
	#pragma omp parallel for schedule(static,1) private(i)
	for (i = 0; i < 4; i++)
	{
		cudaSetDevice(deviceId[i]);
		cudaFree(dev_p[i]);
		cudaFree(dev_v[i]);
	}

	return 0;
}