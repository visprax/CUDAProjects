#ifndef COMMON_H
#define COMMON_H 1

#include <mpi.h>
#include <cuda.h>

#define MPIErrorCheck(call) \
    do { \
        int __ierr = (call); \
        if(__ierr != MPI_SUCCESS) { \
            char err_str[MPI_MAX_ERROR_STRING]; \
            int err_str_len = 0; \
            MPI_Error_string(__ierr, err_str, &err_str_len); \
            if(rank == 0) { \
                fprintf(stderr, "MPI Error: %s:%d %d:%s\n", __FILE__, __LINE__, __ierr, err_str); \
            }
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

#define CUDAErrorCheck(call) \
    do { \
        cudaError_t cuErr = (call); \
        if(cuErr != cudaSuccess) { \
            if(rank == 0) { \
                fprintf(stderr, "CUDA Error: %s:%d %d:%s\n", __FILE__, __LINE__, cuErr, cudaGetErrorString(cuErr)); \
            } \
            MPI_Abort(MPI_COMM_WORLD, 2); \
        } \
    } while(0)

#ifdef __cplucplus
extern "C" {

}
#endif // __cpluscplus

#endif // COMMON_H
