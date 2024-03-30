#ifndef REACTION_DIFFUSION_H
#define REACTION_DIFFUSION_H

#include "automaton.h"

class ReactionDiffusion : public Automaton<float>{

    public:
        float *d_state_u, *d_state_v;
        float *d_nextState_u, *d_nextState_v;

        float dt, du, dv, f, k;

        int tick = 0;

        ReactionDiffusion(int _nx, int _ny, float _dt, float _du, float _dv, float _k, float _f);
        ~ReactionDiffusion();

        virtual Color getColor(float stateVal);
        virtual void update();
};

#endif