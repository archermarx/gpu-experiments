#ifndef AUTOMATON_H
#define AUTOMATON_H

#include <vector>
#include "canvas.h"
#include "color.h"

class Automaton{

    public:
        const unsigned int nx, ny;
        std::vector<std::vector<int>> state;

        Automaton(unsigned int _nx, unsigned int _ny):
            nx(_nx), ny(_ny),
            state(nx, std::vector<int>(ny, 0)) {}

        void draw(Canvas& canvas);
        virtual Color getColor(int stateVal) {return BLACK;}
        virtual void update() {}
};

#endif