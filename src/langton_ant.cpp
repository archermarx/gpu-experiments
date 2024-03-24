#include "langton_ant.h"
#include "utils.h"

LangtonAnt::LangtonAnt(int _nx, int _ny, std::pair<float, float> _pos, Direction _dir)
    : nx(_nx),
      ny(_ny),
      pos((int)(_pos.first * nx), (int)(_pos.second*ny)),
      dir(_dir),
      state(nx, std::vector<uint8_t>(ny, 0)),
      rules({true, false}),
      colors({WHITE, BLACK}) {}

LangtonAnt::LangtonAnt(int _nx, int _ny, std::pair<float, float> _pos, Direction _dir, std::string _rules, std::vector<Color> _colors)
    : nx(_nx),
      ny(_ny),
      pos((int)(_pos.first * nx), (int)(_pos.second*ny)),
      dir(_dir),
      state(nx, std::vector<uint8_t>(ny, 0)),
      rules(_rules.size()),
      colors(_colors) {

        // Parse rule string (L -> false, R -> true, all else -> error)
        for (int i = 0; i < rules.size(); i++) {
            if (_rules[i] == 'L') {
                rules[i] = false;
            } else if (_rules[i] == 'R') {
                rules[i] = true;
            } else {
                std::cout << "Invalid rule character " << _rules[i] << "at position " << i << "in rule string " << _rules << std::endl;
                throw;
            }
        }

    }

void LangtonAnt::update() {
    // get current board state (false -> black or true -> white)
    auto currentState = state[pos.first][pos.second];

    // flip color of square
    state[pos.first][pos.second] = wrap<uint8_t>(currentState + 1, rules.size());

    // rotate ant
    if (rules[currentState]) {
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
            auto color_ind = wrap<uint8_t>(stateVal, colors.size());
            auto color = colors[color_ind];
            canvas.contents[pixelIndex] = color.r;
            canvas.contents[pixelIndex+1] = color.g;
            canvas.contents[pixelIndex+2] = color.b;
            canvas.contents[pixelIndex+3] = 255;
            pixelIndex += 4;
        }
    }
}
