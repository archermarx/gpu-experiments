#include "game_of_life.h"
#include "utils.h"
#include "color.h"

void GameOfLife::update() {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int numNeighbors = countNeighbors(i, j);

            if (state[i][j]) {
                if (numNeighbors < 2 || numNeighbors > 3) {
                    nextState[i][j] = false;
                } else {
                    nextState[i][j] = true;
                }
            } else {
                if (numNeighbors == 3) {
                    nextState[i][j] = true;
                } else {
                    nextState[i][j] = false;
                }
            }
        }
    }

    std::swap(nextState, state);
}

int GameOfLife::countNeighbors(int i, int j) {
    int numNeighbors = 0;
    for (int x_offset = -1; x_offset <= 1; x_offset++) {
        for (int y_offset = -1; y_offset <= 1; y_offset++) {
            if (x_offset == 0 && y_offset == 0) {
                continue;
            }
            int ind_x = wrap<int>(i + x_offset, nx);
            int ind_y = wrap<int>(j + y_offset, ny);

            numNeighbors += state[ind_x][ind_y];
        }
    }

    return numNeighbors;
}

Color GameOfLife::getColor(int state) {
    if (state) {
        return CYAN;
    } else {
        return BLUE;
    }
}