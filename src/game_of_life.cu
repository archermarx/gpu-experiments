#include "game_of_life.h"
#include "utils.h"
#include "color.h"

void GameOfLife::update() {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int numNeighbors = countNeighbors(i, j);

            if (state[index2D(i, j)]) {
                if (numNeighbors < 2 || numNeighbors > 3) {
                    nextState[index2D(i, j)] = false;
                } else {
                    nextState[index2D(i, j)] = true;
                }
            } else {
                if (numNeighbors == 3) {
                    nextState[index2D(i, j)] = true;
                } else {
                    nextState[index2D(i, j)] = false;
                }
            }
        }
    }

    std::swap(nextState, state);
}

int GameOfLife::countNeighbors(int i, int j) {
    int numNeighbors = 0;
    for (int ind_x = i-1; ind_x <= i+1; ind_x++) {
        for (int ind_y = j-1; ind_y <= j+1; ind_y++) {
            if (ind_x == i && ind_y == j) {
                continue;
            }

            numNeighbors += get(ind_x, ind_y);
        }
    }

    return numNeighbors;
}

Color GameOfLife::getColor(bool stateVal) {
    if (stateVal) {
        return CYAN;
    } else {
        return BLUE;
    }
}
