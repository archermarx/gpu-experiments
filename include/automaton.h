#ifndef AUTOMATON_H
#define AUTOMATON_H

#include <stdio.h>
#include <vector>
#include "canvas.h"
#include "color.h"
#include "utils.h"

template <class T>
class Automaton{

    public:
        const unsigned int nx, ny;
        const unsigned int stateSize;
        std::vector<T> state;

        virtual ~Automaton() {}

        Automaton(unsigned int _nx, unsigned int _ny):
            nx(_nx), ny(_ny),
            state(nx * ny, 0),
            stateSize(nx * ny * sizeof(T)) {}

        T get(unsigned int i, unsigned int j) {
            return state[wrapIndex2D(i, j, nx, ny)];
        }

        void set(unsigned int i, unsigned int j, T stateVal){
            state[wrapIndex2D(i, j, nx, ny)] = stateVal;
        }

        void draw(Canvas& canvas) {
            int pixelIndex = 0;
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    T stateVal = state[wrapIndex2D(i, j, nx, ny)];
                    auto color = getColor(stateVal);
                    canvas.contents[pixelIndex  ] = color.r;
                    canvas.contents[pixelIndex+1] = color.g;
                    canvas.contents[pixelIndex+2] = color.b;
                    canvas.contents[pixelIndex+3] = 255;
                    pixelIndex += 4;
                }
            }
        }

        virtual Color getColor(T stateVal) = 0;
        virtual void update() = 0;
};

#endif