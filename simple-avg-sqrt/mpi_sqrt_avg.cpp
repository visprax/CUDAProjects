#include <iostream>
#include <mpi.h>

/*
 * Root MPI process creates a array, initializes it.
 * We scatter the array to available MPI processes, 
 * Each MPI process launches a GPU kernel to compute 
 * the square root of each element of the array, for which
 * it needs to copy the data per MPI process in the Host to 
 * Device and back to the Host after computation. Then we
 * MPI_Reduce each sum into a global sum and report the average.
 */

#define MPIErrorCheck(call) \
    do { \
        if(call != MPI_SUCCESS) { \
            fprintf(stderr, "MPI Error: %s:%d %s", __FILE__, __LINE__, )

int main(int argc, char** argv) {

}
