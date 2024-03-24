#include "langton_ant.h"
#include "utils.h"

LangtonAnt::LangtonAnt(int _nx, int _ny, std::pair<int, int> _pos, Direction _dir)
    : nx(_nx),
      ny(_ny),
      pos(_pos),
      dir(_dir),
      state(nx, std::vector<bool>(ny, true)) {}

void LangtonAnt::update() {
    // get current board state (false -> black or true -> white)
    bool currentState = state[pos.first][pos.second];

    // flip color of square
    state[pos.first][pos.second] = !currentState;

    // rotate ant
    if (currentState) {
        dir -= 1;
    } else {
        dir += 1;
    }

    // wrap direction
    dir = wrap<int8_t>(dir, 4);

    // move ant, wrapping on boundaries
    auto& direction = directions[dir];
    pos.first  = wrap<int>(pos.first + direction.first, nx);
    pos.second = wrap<int>(pos.second + direction.second, ny);
}

void LangtonAnt::draw(Canvas& canvas) {
    int pixelIndex = 0;
    for (auto& row: state) {
        for (auto stateVal: row) {
            int color = stateVal ? 255 : 0;
            canvas.contents[pixelIndex] = color;
            canvas.contents[pixelIndex+1] = color;
            canvas.contents[pixelIndex+2] = color;
            canvas.contents[pixelIndex+3] = 255;
            pixelIndex += 4;
        }
    }
}
