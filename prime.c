#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void sieve(int n, int rank, int size) {
    // Each process handles a portion of the range from 2 to n
    int local_start = 2 + rank * (n / size);
    int local_end = (rank == size - 1) ? n : local_start + (n / size) - 1;

    // Ensure local_start is at least 2
    if (local_start < 2) local_start = 2;

    // Allocate array to track prime numbers
    int range_size = local_end - local_start + 1;
    int* is_prime = malloc(range_size * sizeof(int));
    for (int i = 0; i < range_size; i++) {
        is_prime[i] = 1; // Assume all are prime initially
    }

    // Sieve process
    for (int i = 2; i <= sqrt(n); i++) {
        // Each process marks its own multiples
        // Calculate the first multiple of i in the local range
        int first_multiple = (local_start / i) * i;
        if (first_multiple < local_start) {
            first_multiple += i;
        }
        if (first_multiple == i) first_multiple += i; // Avoid marking the prime itself

        for (int j = first_multiple; j <= local_end; j += i) {
            if (j >= local_start) {
                is_prime[j - local_start] = 0; // Mark as non-prime
            }
        }
    }

    // Count local primes and print them
    printf("Process %d found primes: ", rank);
    for (int i = 0; i < range_size; i++) {
        if (is_prime[i]) {
            printf("%d ", local_start + i);
        }
    }
    printf("\n");

    // Gather results from all processes
    int* global_primes = NULL;
    if (rank == 0) {
        global_primes = malloc(n * sizeof(int));
    }

    // Count local primes
    int local_prime_count = 0;
    for (int i = 0; i < range_size; i++) {
        if (is_prime[i]) {
            local_prime_count++;
        }
    }

    // Gather counts of local primes
    int* counts = malloc(size * sizeof(int));
    MPI_Gather(&local_prime_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // On root, determine total number of primes
    int total_primes = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            total_primes += counts[i];
        }
    }

    // Allocate space for global primes
    if (rank == 0) {
        global_primes = realloc(global_primes, total_primes * sizeof(int));
    }

    // Gather actual prime numbers
    int* displacements = NULL;
    if (rank == 0) {
        displacements = malloc(size * sizeof(int));
        int displacement = 0;
        for (int i = 0; i < size; i++) {
            displacements[i] = displacement;
            displacement += counts[i];
        }
    }

    // Use Gatherv to collect all primes from each process
    MPI_Gatherv(is_prime, local_prime_count, MPI_INT, global_primes, counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

    // Output results from root process
    if (rank == 0) {
        printf("Total primes found: %d\n", total_primes);
        printf("All primes up to %d:\n", n);
        for (int i = 0; i < total_primes; i++) {
            printf("%d ", global_primes[i]);
        }
        printf("\n");
        free(global_primes);
        free(displacements);
    }

    free(is_prime);
    free(counts);
}

int main(int argc, char** argv) {
    int rank, size;
    int n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the upper limit n: ");
        scanf("%d", &n);
    }

    // Broadcast the value of n to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);


    double start = MPI_Wtime();
    sieve(n, rank, size);
    double end = MPI_Wtime();
    
    if (rank == 0) {
        printf("Execution Time: %.6f seconds\n", end - start);
    }

    MPI_Finalize();
    return 0;
}
