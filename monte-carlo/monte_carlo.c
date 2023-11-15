#include <mpi.h>
#include <stdio.h>

#include "monte_carlo.h"
#include "util.h"

double monte_carlo(double *xs, double *ys, int num_points, int mpi_rank, int mpi_world_size, int threads_per_process) {
  int count = 0;

  // TODO: Parallelize the code using mpi_world_size processes (1 process per
  // node.
  // In total, (mpi_world_size * threads_per_process) threads will collaborate
  // to compute pi.

  MPI_Bcast(xs, num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(ys, num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  #pragma omp parallel num_threads(threads_per_process)
  {
    #pragma omp for reduction(+:count)
    for (int i = mpi_rank; i < num_points; i += mpi_world_size) {
      double x = xs[i];
      double y = ys[i];

      if (x * x + y * y <= 1) count++;
    }
  }

  int total_count = 0;
  MPI_Reduce(&count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Rank 0 should return the estimated PI value
  // Other processes can return any value (don't care)
  return (double) 4 * total_count / num_points;
}
