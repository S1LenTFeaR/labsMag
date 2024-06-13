#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define BLOCK_SIZE 16
#define N (BLOCK_SIZE * 16)

struct particle { float x, y, z; };

int main(int argc, char* argv[])
{
	char filename[50];

	struct particle* p = (particle*)malloc(sizeof(*p) * N);	// Положение частиц (x, y, z)
	struct particle* f = (particle*)malloc(sizeof(*f) * N);	// Сила, действующая на каждую частицу (x, y, z)
	struct particle* v = (particle*)malloc(sizeof(*v) * N);	// Скорость частицы (x, y, z)
	float* m = (float*)malloc(sizeof(*m) * N);
	sprintf(filename, "data\\in_N%d.txt", N);
	FILE* gen = fopen(filename, "w");
	fprintf(gen, "%d %d\n", BLOCK_SIZE, N);
	for (int i = 0; i < N; i++)
	{
		p[i].x = (rand() / (float)RAND_MAX - 0.5) * 200; // -100 , 100
		p[i].y = (rand() / (float)RAND_MAX - 0.5) * 200;
		p[i].z = (rand() / (float)RAND_MAX - 0.5) * 200;
		v[i].x = (rand() / (float)RAND_MAX - 0.5) * 20; // -10 , 10
		v[i].y = (rand() / (float)RAND_MAX - 0.5) * 20;
		v[i].z = (rand() / (float)RAND_MAX - 0.5) * 20;
		m[i] = rand() / (float)RAND_MAX * 100 + 0.01; // 0.01 , 100.01
		f[i].x = f[i].y = f[i].z = 0;
		fprintf(gen, "%f %f %f %f %f %f %f\n", p[i].x, p[i].y, p[i].z, v[i].x, v[i].y, v[i].z, m[i]);
	}
	fclose(gen);

}