#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include <vector>
#include "canvas.h"
#include "automaton.h"
#include "cuda_helpers.h"

class GameOfLife : public Automaton<char>{
    public:
        std::vector<char> nextState;

        char *d_state;
        char *d_nextState;
        int frame = 0;

        GameOfLife(int _nx, int _ny);
        ~GameOfLife();

        int countNeighbors(int i, int j);
        virtual Color getColor(char stateVal);
        virtual void update();
};

#endif