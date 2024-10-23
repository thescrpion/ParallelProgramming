#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void sieve_of_eratosthenes(int n, int rank, int size){

    int *is_prime = malloc((n + 1) * sizeof(int));
    
    for (int i = 0; i <= n; i++) {
        is_prime[i] = 1; 
    }
    
    is_prime[0] = is_prime[1] = 0; 

    int target = (int)sqrt(n);
    
    for (int p = 2; p <= target; p++){
    
        if (is_prime[p]){
        
            for (int multiple = p * p; multiple <= n; multiple += p) {
                
                   is_prime[multiple] = 0;
                 
            }
        }
    }

    int *global_is_prime = NULL;
    
    if (rank == 0) {
        global_is_prime = malloc((n + 1) * sizeof(int));
    }
    

    
    MPI_Gather(is_prime, n + 1, MPI_INT, global_is_prime, n + 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) { 
    
        printf("Prime numbers up to %d: ", n);
        
        for (int i = 2; i <= n; i++) {
        
            if (global_is_prime[i]) {
                printf("%d ", i); 
            }
            
        }
        
        printf("\n");
        free(global_is_prime);
        
        
    }
    
    
    free(is_prime);

    
}

int main(int argc, char *argv[]) {

    int n;
    

    MPI_Init(&argc, &argv); 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    if (rank == 0) {
        printf("Enter the upper limit n for parallel execution: ");
        scanf("%d", &n); 
    }

    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime(); 
    sieve_of_eratosthenes(n, rank, size); 
    double end_time = MPI_Wtime(); 

    if (rank == 0) { 
        double elapsed_time = end_time - start_time; 
        printf("MPI execution time: %f seconds\n", elapsed_time);  
    }

   
    MPI_Finalize(); 
    return 0;
}

