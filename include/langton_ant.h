#ifndef LANGTON_ANT_H
#define LANGTON_ANT_H

#include <cstdint>
#include <utility>
#include <vector>
#include "canvas.h"
#include "color.h"
#include "automaton.h"

using Direction = int8_t;

const Direction EAST = 0;
const Direction NORTH = 1;
const Direction WEST = 2;
const Direction SOUTH = 3;

const std::vector<std::pair<int, int>>
directions({
    std::make_pair( 1,  0),   // east
    std::make_pair( 0,  1),   // north
    std::make_pair(-1,  0),   // west
    std::make_pair( 0, -1)    // south
});

class LangtonAnt : public Automaton<uint8_t>{
    public:
        std::pair<int, int> pos;
        Direction dir;
        std::vector<bool> rules;
        std::vector<Color> colors;

        LangtonAnt(int _nx, int _ny, std::pair<float, float> _pos, Direction _dir);
        LangtonAnt(int _nx, int _ny, std::pair<float, float> _pos, Direction _dir, std::string rules, std::vector<Color> colors);

        virtual void update();
        virtual Color getColor(uint8_t stateVal);
};

#endif