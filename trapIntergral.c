#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Function to evaluate the curve (y = f(x))
float f(float x) {
    return x * x; // Example: y = x^2
}

// Function to compute the area of a trapezoid (sequential version)
float trapezoid_area_sequential(float a, float b, int n) {
    float area = 0.0f;
    float d = (b - a) / n;

    for (int i = 0; i < n; i++) {
        float x = a + i * d;
        area += f(x) + f(x + d);
    }

    return area * d / 2.0f;
}

// Function to compute the area of a trapezoid (parallel version)
float trapezoid_area_parallel(float a, float b, int n, int rank, int size) {
    float d = (b - a) / n; // delta
    float region = (b - a) / size;

    // Calculate local bounds for each process
    float start = a + rank * region;
    float end = start + region;
    float local_area = 0.0f;

    for (int i = 0; i < n / size; i++) {
        float x = start + i * d;
        local_area += f(x) + f(x + d);
    }

    return local_area * d / 2.0f;
}

int main(int argc, char** argv) {
    int rank, size;
    float a = 0.0f, b = 1.0f; // Limits of integration
    int n;
    float total_area;
    double sequential_time;

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    if (rank == 0) {
        // Get the number of intervals from the user
        printf("Enter the number of intervals: ");
        scanf("%d", &n);

        // Measure execution time for the sequential version
        clock_t start_time = clock();
        float sequential_area = trapezoid_area_sequential(a, b, n);
        clock_t end_time = clock();
        sequential_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("The total area under the curve (sequential) is: %f\n", sequential_area);
        printf("Execution time (sequential): %f seconds\n", sequential_time);
    }

    // Broadcast the number of intervals to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Start timing for the parallel version
    double start_time = MPI_Wtime();
    
    // Each process calculates the area of its subinterval
    float local_area = trapezoid_area_parallel(a, b, n, rank, size);
    
    // Reduce all local areas to the total area on the root process
    MPI_Reduce(&local_area, &total_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // End timing for the parallel version
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        
        double parallel_time = end_time - start_time;
        printf("The total area under the curve (parallel) is: %f\n", total_area);
        printf("Execution time (parallel): %f seconds\n", parallel_time);
        
        // Calculate and print speedup
        double speedup = sequential_time / parallel_time;
        printf("Speedup factor: %f\n", speedup);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}

