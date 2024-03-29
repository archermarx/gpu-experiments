#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include <vector>
#include "canvas.h"
#include "automaton.h"

class GameOfLife : public Automaton<bool>{
    public:
        std::vector<bool> nextState;

        GameOfLife(int _nx, int _ny) :
            Automaton<bool>(_nx, _ny),
            nextState(_nx * _ny, 0){}

        int countNeighbors(int i, int j);
        virtual Color getColor(bool stateVal);
        virtual void update();
};

#endif