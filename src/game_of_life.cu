#include "game_of_life.h"
#include "color.h"

const int BLOCK_SIZE_X = 32;
const int BLOCK_SIZE_Y = 32;

 GameOfLife::GameOfLife(int _nx, int _ny):
    Automaton<char>(_nx, _ny), nextState(_nx * _ny, 0){

    // Allocate CUDA arrays
    CUDA_CHECK( cudaMalloc((void**) &d_state,     stateSize) );
    CUDA_CHECK( cudaMalloc((void**) &d_nextState, stateSize) );
}

__global__
void kernel_gameOfLife(char *nextState, const char *state, const int nx, const int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < nx && j < ny) {
        int c  = wrapIndex2D(i, j, nx, ny);
        int n  = wrapIndex2D(i, j+1, nx, ny);
        int s  = wrapIndex2D(i, j-1, nx, ny);
        int e  = wrapIndex2D(i+1, j, nx, ny);
        int w  = wrapIndex2D(i-1, j, nx, ny);
        int se  = wrapIndex2D(i+1, j-1, nx, ny);
        int sw  = wrapIndex2D(i-1, j-1, nx, ny);
        int ne  = wrapIndex2D(i+1, j+1, nx, ny);
        int nw  = wrapIndex2D(i-1, j+1, nx, ny);

        int numNeighbors = state[n] + state[ne] + state[nw] + state[s] + state[se] + state[sw] + state[w] + state[e];
        bool alive = state[c] == 1;
        nextState[c] = (alive && (numNeighbors == 2 || numNeighbors == 3)) || (!alive && numNeighbors == 3);
    }
}

void GameOfLife::update() {

    // copy state to GPU
    if (frame == 0) {
        cudaMemcpy( d_state, state.data(), stateSize, cudaMemcpyHostToDevice);
    }

    // launch kernel
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y,1);
    dim3 gridSize(nx/BLOCK_SIZE_X, ny/BLOCK_SIZE_Y, 1);

    kernel_gameOfLife<<<gridSize, blockSize>>>(d_nextState, d_state, nx, ny);

    // swap pointers
    std::swap(d_nextState, d_state);

    // copy state back from GPU
    cudaMemcpy( state.data(), d_state, stateSize, cudaMemcpyDeviceToHost);
}

Color GameOfLife::getColor(char stateVal) {
    if (stateVal) {
        return CYAN;
    } else {
        return BLUE;
    }
}

GameOfLife::~GameOfLife() {
    // deallocate cuda arrays
    CUDA_CHECK( cudaFree(d_state)     );
    CUDA_CHECK( cudaFree(d_nextState) );
}
