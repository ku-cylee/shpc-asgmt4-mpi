#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE   32
#define VECTOR_SIZE 16

void matmul(float *A, float *B, float *C, int M, int N, int K,
    int threads_per_process, int mpi_rank, int mpi_world_size) {
  int i, j, jj, k, kk, v;
  int M_per_rank = M / mpi_world_size;
  int M_start = M_per_rank * mpi_rank;
  int VEC_K = K / VECTOR_SIZE;

  MPI_Scatter(
    A, M * K / mpi_world_size, MPI_FLOAT,
    &A[M * K / mpi_world_size * mpi_rank], M * K / mpi_world_size, MPI_FLOAT,
    0, MPI_COMM_WORLD);
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  __m512 *A_vecs = (__m512 *)aligned_alloc(32, M_per_rank * VEC_K * sizeof(__m512));
  __m512 *B_vecs = (__m512 *)aligned_alloc(32, VEC_K * N * sizeof(__m512));

  #pragma omp parallel num_threads(threads_per_process) shared(i)
  {
    #pragma omp for
    for (i = 0; i < M_per_rank * VEC_K; i++) {
      A_vecs[i] = _mm512_load_ps(&A[M_start * K + i * VECTOR_SIZE]);
    }
  }

  #pragma omp parallel num_threads(threads_per_process) shared(j)
  {
    #pragma omp for private(jj, k, v)
    for (j = 0; j < N; j += TILE_SIZE) {
      for (jj = j; jj < j + TILE_SIZE; jj++) {
        for (k = 0; k < VEC_K; k++) {
          float b_[VECTOR_SIZE];
          for (v = 0; v < VECTOR_SIZE; v++) b_[v] = B[(k * VECTOR_SIZE + v) * N + jj];
          B_vecs[jj * VEC_K + k] = _mm512_load_ps(b_);
        }
      }
    }
  }

  __m512 a, b, c;
  #pragma omp parallel num_threads(threads_per_process) shared(j)
  {
    #pragma omp for private(i, jj, k, kk, a, b, c)
    for (j = 0; j < N; j += TILE_SIZE) {
      for (k = 0; k < VEC_K; k += TILE_SIZE) {
        for (i = 0; i < M_per_rank; i++) {
          for (jj = j; jj < j + TILE_SIZE; jj++) {
            c = _mm512_setzero_ps();
            for (kk = k; kk < k + TILE_SIZE; kk++) {
              a = A_vecs[i * VEC_K + kk];
              b = B_vecs[jj * VEC_K + kk];
              c = _mm512_fmadd_ps(a, b, c);
            }
            C[(i + M_start) * N + jj] += _mm512_reduce_add_ps(c);
          }
        }
      }
    }
  }

  free(A_vecs);
  free(B_vecs);

  MPI_Gather(
    &C[M / mpi_world_size * mpi_rank * N], M * N / mpi_world_size, MPI_FLOAT,
    C, M * N / mpi_world_size, MPI_FLOAT,
    0, MPI_COMM_WORLD);
}
