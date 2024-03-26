#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include <vector>
#include "canvas.h"

class GameOfLife {
    public:
        const int nx, ny;
        std::vector<std::vector<bool>> state;
        std::vector<std::vector<bool>> nextState;

        GameOfLife(int _nx, int _ny) :
            nx(_nx), ny(_ny),
            state(nx, std::vector<bool>(ny, 0)),
            nextState(nx, std::vector<bool>(ny, 0)){}

        int countNeighbors(int i, int j);
        void update();
        void draw(Canvas& canvas);
};

#endif