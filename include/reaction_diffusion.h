#ifndef REACTION_DIFFUSION_H
#define REACTION_DIFFUSION_H

#include "automaton.h"

class ReactionDiffusion : public Automaton<float>{

    public:

        float *d_state, *d_nextState;

        ReactionDiffusion(int _nx, int _ny);
        ~ReactionDiffusion();

        virtual Color getColor(char stateVal);
        virtual void update();
};

#endif