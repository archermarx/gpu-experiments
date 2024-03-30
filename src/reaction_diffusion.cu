#include <random>
#include <algorithm>
#include "reaction_diffusion.h"
#include "cuda_helpers.h"
#include "color.h"

ReactionDiffusion::ReactionDiffusion(int _nx, int _ny, float _dt, float _du, float _dv, float _k, float _f) :
    Automaton<float>(_nx, _ny), dt(_dt), du(_du), dv(_dv), f(_f), k(_k) {

    // Allocate CUDA arrays
    CUDA_CHECK( cudaMalloc((void**) &d_state_u,     stateSize) );
    CUDA_CHECK( cudaMalloc((void**) &d_state_v,     stateSize) );
    CUDA_CHECK( cudaMalloc((void**) &d_nextState_u, stateSize) );
    CUDA_CHECK( cudaMalloc((void**) &d_nextState_v, stateSize) );
}

Color ReactionDiffusion::getColor(float stateVal) {
    int grey = (int) std::clamp((stateVal - 0.3f) / 0.7f * 255.0f, 0.0f, 255.0f);
    return Color(grey, grey, grey);
}

__global__
void kernel_diffuse(float *nextState, float *state, float D, float dt, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float dy = 2.5f / ny;
    float dx = 2.5f / nx;

    float x_scale = 1.0f / (dx * dx);
    float y_scale = 1.0f / (dy * dy);

    if (i < nx && j < ny) {
        int id = wrapIndex2D(i, j, nx, ny);
        int n  = wrapIndex2D(i, j+1, nx, ny);
        int s  = wrapIndex2D(i, j-1, nx, ny);
        int e  = wrapIndex2D(i+1, j, nx, ny);
        int w  = wrapIndex2D(i-1, j, nx, ny);
        nextState[id] = state[id] + dt * D * (
            (state[e] + state[w] - 2 * state[id]) * x_scale +
            (state[n] + state[s] - 2 * state[id]) * y_scale
        );
    }
}

__global__
void kernel_react(float *U, float*V, float k, float f, float dt, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < nx && j < ny) {
        int id = wrapIndex2D(i, j, nx, ny);
        float u = U[id];
        float v = V[id];
        U[id] += dt * (-u*v*v + f * (1 - u));
        V[id] += dt * (u*v*v - (f + k) * v);
    }
}

void ReactionDiffusion::update() {

    if (tick == 0) {
        // Create initial condition
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-0.005, 0.005);
        std::vector<float> conc_v(nx * ny, 0);

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int id = i + j * nx;
                if (float(i) / nx > 0.45 && float(i) / nx < 0.55 && float(j) / ny > 0.45 && float(j) / ny < 0.55) {
                    state[id] = 0.5;
                    conc_v[id] = 0.25;
                } else {
                    state[id] = 1.0;
                    conc_v[id] = 0.0;
                }
                state[id] *= (1 + distribution(generator));
                conc_v[id] *= (1 + distribution(generator));
            }
        }

        std::cout << "System initialized." << std::endl;

        // Copy initial condition to GPU
        CUDA_CHECK( cudaMemcpy(d_state_u, state.data(),  stateSize, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(d_state_v, conc_v.data(), stateSize, cudaMemcpyHostToDevice) );
    }

    const int BLOCK_SIZE_X = 32;
    const int BLOCK_SIZE_Y = 32;

    // Perform diffusion of both components
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 gridSize(nx / BLOCK_SIZE_X, ny / BLOCK_SIZE_Y, 1);
    kernel_diffuse<<<gridSize, blockSize>>>(d_nextState_u, d_state_u, du, dt, nx, ny);
    kernel_diffuse<<<gridSize, blockSize>>>(d_nextState_v, d_state_v, dv, dt, nx, ny);

    // swap previous and new states
    std::swap(d_state_u, d_nextState_u);
    std::swap(d_state_v, d_nextState_v);

    // Reactions
    kernel_react<<<gridSize, blockSize>>>(d_state_u, d_state_v, k, f, dt, nx, ny);

    // Copy GPU memory to CPU
    CUDA_CHECK( cudaMemcpy(state.data(), d_state_u, stateSize, cudaMemcpyDeviceToHost) );
    tick += 1;
}

ReactionDiffusion::~ReactionDiffusion() {
    CUDA_CHECK( cudaFree(d_state_u) );
    CUDA_CHECK( cudaFree(d_state_v) );
    CUDA_CHECK( cudaFree(d_nextState_u) );
    CUDA_CHECK( cudaFree(d_nextState_v) );
}

