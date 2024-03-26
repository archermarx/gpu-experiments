#ifndef AUTOMATON_H
#define AUTOMATON_H

#include <vector>
#include "canvas.h"
#include "color.h"

template <class T>
class Automaton{

    public:
        const unsigned int nx, ny;
        std::vector<std::vector<T>> state;

        virtual ~Automaton() {}

        Automaton(unsigned int _nx, unsigned int _ny):
            nx(_nx), ny(_ny),
            state(nx, std::vector<T>(ny, 0)) {}

        T get(unsigned int i, unsigned int j);
        void set(unsigned int i, unsigned int j, T stateVal);

        void draw(Canvas& canvas);

        virtual Color getColor(T stateVal) = 0;
        virtual void update() = 0;
};

#include "automaton.h"
#include "utils.h"

template <typename T>
void Automaton<T>::draw(Canvas& canvas) {
    int pixelIndex = 0;
    for (auto& row: state) {
        for (T stateVal: row) {
            auto color = getColor(stateVal);
            canvas.contents[pixelIndex] = color.r;
            canvas.contents[pixelIndex+1] = color.g;
            canvas.contents[pixelIndex+2] = color.b;
            canvas.contents[pixelIndex+3] = 255;
            pixelIndex += 4;
        }
    }
}

template <typename T>
T Automaton<T>::get(unsigned int i, unsigned int j) {
    return state[wrap(i, nx)][wrap(j, ny)];
}

template <typename T>
void Automaton<T>::set(unsigned int i, unsigned int j, T stateVal){
    state[wrap(i, nx)][wrap(j, ny)] = stateVal;
}

#endif