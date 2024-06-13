#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define BLOCK_SIZE 8 // количество потоков в каждом блоке
#define N 2048        // размер матрицы N*N

__global__ void matMult(int *a, int *b, int *_c, int i)
{
    int index = threadIdx.x * N + blockIdx.y;
    int c = 0;
    int sum = 0;
    for (int k = 0; k < N; k++)
    {
        // Вычисляем сумму произведений элементов a и b
        c += a[((i * N / 4) + threadIdx.x) * N + k] * b[blockIdx.y + k * N];
    }
    _c[index] = c;
}

void matMultCPU(int *a, int *b, int n, int *c)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main(int argc, char *argv[])
{
    int numBytes = N * N * sizeof(int);
    int nthr, i, j, it, i2, i3, i4;
    int *a = new int[N * N];
    int *b = new int[N * N];
    int *c = new int[N * N];
    int *resCPU = new int[N * N];
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int k = N * i + j;
            a[k] = rand() % 10;
            b[k] = rand() % 10;
        }
    }
    // Выделение памяти на девайсе
    int *adev = NULL;
    int *bdev = NULL;
    int *cdev[4] = {NULL, NULL, NULL, NULL};

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
#pragma omp parallel for schedule(static, 1) private(i, j)
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

// Выделение памяти для массивов GPUs
#pragma omp parallel for schedule(static, 1) private(i)
    for (i = 0; i < 4; i++)
    {
        // Использование видеокарты i
        cudaSetDevice(deviceId[i]);
        // p и v
        cudaMalloc((void **)&cdev[i], numBytes / 4);
        // Копирование i четверти массива p на видеокрту i
        cudaMemcpy(cdev[i], c + i * (N / 4 * N), (numBytes / 4), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void **)&adev, numBytes);
    cudaMalloc((void **)&bdev, numBytes);

    // Копирование данных с хоста на девайс
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    // Видеокарта 1
    cudaSetDevice(deviceId[1]);
    cudaEvent_t start, stop;
    float timeGPU = 0.0f;
    // Создание событий для замера времени
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    #pragma omp parallel for schedule(static, 1) private(i, j, i2, i3, i4)
    for (i = 0; i < 4; i++)
    {
        cudaSetDevice(deviceId[i]);
        // Прямой доступ к выделенной памяти остальных GPU
        for (j = 0; j < 4; j++)
        {
            if (j != i)
                cudaDeviceEnablePeerAccess(deviceId[j], 0);
        }
        // Индексация частей массивов old с других GPU
        if (i == 0)
        {
            i2 = 1;
            i3 = 2;
            i4 = 3;
        }
        else if (i == 1)
        {
            i2 = 0;
            i3 = 3;
            i4 = 2;
        }
        else if (i == 2)
        {
            i2 = 3;
            i3 = 1;
            i4 = 0;
        }
        else if (i == 3)
        {
            i2 = 2;
            i3 = 0;
            i4 = 1;
        }
        // Перемещение частиц
        matMult<<<dim3(1, N), dim3(N / 4, 1)>>>(adev, bdev, cdev[i], i);
    }
    #pragma omp parallel for schedule(static,1) private(i)
    for (i = 0; i < 4; i++) 
    {
        cudaSetDevice(deviceId[i]);
        // Копирование расчитанных значений со всех GPU на CPU
        cudaMemcpy(c + i * (N / 4 * N), cdev[i], numBytes / 4, cudaMemcpyDeviceToHost);
    }
    cudaSetDevice(deviceId[1]);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeGPU, start, stop);
    // Вывод времени на экран
	printf("# Elapsed time GPU (sec): %.3f s\n", timeGPU / 1000);
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
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    #pragma omp parallel for schedule(static,1) private(i)
	for (i = 0; i < 4; i++)
	{
		cudaSetDevice(deviceId[i]);
		cudaFree(cdev[i]);
	}
    if(N < 32)
    {
        printf("Матрица c (GPU):\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf("%4d ", c[N * i + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    

    


    float timeCPU = 0.0f;
    // Создание событий для замера времени
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    matMultCPU(a, b, N, resCPU);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeCPU, start, stop);
    // Вывод времени на экран
	printf("# Elapsed time CPU (sec): %.3f s\n", timeCPU / 1000);

    if(N < 32)
    {
        printf("Матрица c (CPU):\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf("%4d ", resCPU[N * i + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    int isEqual = true;
    for(int i; i < N * N; i++)
    {
        if(c[i] != resCPU[i])
        {
            printf("Matix not equal:\n");
            isEqual = false;
            break;
        }     
    }
    if(isEqual)
        printf("Matix is equal:\n");
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] resCPU;
    return 0;
}