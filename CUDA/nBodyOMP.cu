#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define G 6.67e-11
#define dt 1e-3

struct particle { float x, y, z; };

void calculate_forces(struct particle* p, struct particle* f, float* m, int n)
{
	#pragma omp for schedule(dynamic, omp_get_num_threads()) nowait // Циклическое распределение итераций
	for (int i = 0; i < n - 1; i++)
	{
		for (int j = i + 1; j < n; j++)
		{
			// Вычисление силы, действующей на тело i со стороны j
			float dist = sqrtf(powf(p[i].x - p[j].x, 2) + powf(p[i].y - p[j].y, 2) + powf(p[i].z - p[j].z, 2));
			float mag = (G * m[i] * m[j]) / powf(dist, 2);
			struct particle dir =
			{
				dir.x = p[j].x - p[i].x,
				dir.y = p[j].y - p[i].y,
				dir.z = p[j].z - p[i].z
			};

			// Сумма сил, действующих на тело i
			#pragma omp atomic
			f[i].x += mag * dir.x / dist;
			#pragma omp atomic
			f[i].y += mag * dir.y / dist;
			#pragma omp atomic
			f[i].z += mag * dir.z / dist;
			// Сумма сил, действующих на тело j (симметричность)
			#pragma omp atomic
			f[j].x -= mag * dir.x / dist;
			#pragma omp atomic
			f[j].y -= mag * dir.y / dist;
			#pragma omp atomic
			f[j].z -= mag * dir.z / dist;
		}
	}
}

void move_particles(struct particle* p, struct particle* f, struct particle* v, float* m, int n)
{
	#pragma omp for nowait
	for (int i = 0; i < n; i++)
	{
		struct particle dv =
		{
			dv.x = f[i].x / m[i] * dt,
			dv.y = f[i].y / m[i] * dt,
			dv.z = f[i].z / m[i] * dt,
		};
		struct particle dp =
		{
			dp.x = (v[i].x + dv.x / 2) * dt,
			dp.y = (v[i].y + dv.y / 2) * dt,
			dp.z = (v[i].z + dv.z / 2) * dt,
		};
		v[i].x += dv.x;
		v[i].y += dv.y;
		v[i].z += dv.z;
		p[i].x += dp.x;
		p[i].y += dp.y;
		p[i].z += dp.z;
		f[i].x = f[i].y = f[i].z = 0;
	}
}

int main(int argc, char* argv[])
{
	char filename[50];
	int i, n;
	int n_mass[7] = { 256, 4096, 16384, 65536, 131072, 262144, 524288 };

	printf("Select array: \n");
	for (i = 0; i < 7; i++)
		printf("%d: n = %d\n", i + 1, n_mass[i]);
	scanf("%d", &i);
	if (i >= 1 && i <= 7)
		n = n_mass[i - 1];
	else
		return 0;

	struct particle* p = (particle*)malloc(sizeof(*p) * n);	// Положение частиц (x, y, z)
	struct particle* f = (particle*)malloc(sizeof(*f) * n);	// Сила, действующая на каждую частицу (x, y, z)
	struct particle* v = (particle*)malloc(sizeof(*v) * n);	// Скорость частицы (x, y, z)
	float* m = (float*)malloc(sizeof(*m) * n);				// Масса частицы
	
	sprintf(filename, "data\\in_N%d.txt", n);
	FILE* file = fopen(filename, "r");
	fscanf(file, "%d", &n); fscanf(file, "%d", &n);
	for (i = 0; i < n; i++)
	{
		fscanf(file, "%f%f%f%f%f%f%f", &p[i].x, &p[i].y, &p[i].z, &v[i].x, &v[i].y, &v[i].z, &m[i]);
		f[i].x = f[i].y = f[i].z = 0;
	}

	printf("\nNBody moving in progress (n = %d)...\n\n", n);

	// Замер времени
	cudaEvent_t start, stop;
	float timeCPU = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	#pragma omp parallel //num_threads(8) // Параллельный регион активируется 1 раз
	{
		for (int it = 0; it < 2; it++) // Цикл по времени (модельному)
		{
			calculate_forces(p, f, m, n); // Вычисление сил - О(N^2)
			#pragma omp barrier // Ожидание завершения расчетов f[i]

			move_particles(p, f, v, m, n); // Перемещение тел 0(N)
			#pragma omp barrier // Ожидание завершения обновления p[i], f[i]
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeCPU, start, stop);
	// Вывод времени на экран
	printf("# NBody (n=%d)\n", n);
	printf("# Elapsed time CPU (sec): %.3f s\n", timeCPU / 1000);

	if (filename)
	{
		sprintf(filename, "data\\outOMP_N%d.csv", n);
		FILE* fout = fopen(filename, "w");
		if (!fout)
		{
			fprintf(stderr, "Can't save file/n");
			exit(EXIT_FAILURE);
		}
		fprintf(fout, "x,y,z,m\n");
		for (int i = 0; i < n; i++)
		{
			fprintf(fout, "%15f,%15f,%15f,%15f\n", p[i].x, p[i].y, p[i].z, m[i]);
		}
		fclose(fout);
	}

	free(m);
	free(v);
	free(f);
	free(p);
	return 0;

}