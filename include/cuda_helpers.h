#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define CUDA_CHECK( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif