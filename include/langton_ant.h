#ifndef LANGTON_ANT_H
#define LANGTON_ANT_H

#include <cstdint>
#include <utility>
#include <vector>

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

class LangtonAnt {
    public:
        const int nx;
        const int ny;
        std::pair<int, int> pos;
        Direction dir;
        std::vector<std::vector<bool>> state;

        LangtonAnt(int _nx, int _ny, std::pair<int, int> _pos, Direction _dir);
        void update();
        void draw(std::vector<uint8_t>&);
};

#endif