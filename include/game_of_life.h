#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include <vector>
#include "canvas.h"
#include "automaton.h"

class GameOfLife : public Automaton{
    public:
        std::vector<std::vector<int>> nextState;

        GameOfLife(int _nx, int _ny) :
            Automaton(_nx, _ny),
            nextState(_nx, std::vector<int>(_ny, 0)){}

        int countNeighbors(int i, int j);
        void update();
        void draw(Canvas& canvas);
        Color getColor(int stateVal);
};

#endif