#include "reaction_diffusion.h"
#include "cuda_helpers.h"

ReactionDiffusion::ReactionDiffusion(int _nx, int _ny) :
    Automaton<float>(_nx, _ny) {

    // Allocate CUDA arrays
    CUDA_CHECK( cudaMalloc((void**) &d_state,     stateSize) );
    CUDA_CHECK( cudaMalloc((void**) &d_nextState, stateSize) );
}

ReactionDiffusion::~ReactionDiffusion() {
    CUDA_CHECK( cudaFree(d_state));
    CUDA_CHECK( cudaFree(d_nextState));
}

