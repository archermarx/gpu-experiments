#include "langton_ant.h"
#include "utils.h"

LangtonAnt::LangtonAnt(int _nx, int _ny, std::pair<float, float> _pos, Direction _dir)
    : Automaton<uint8_t>(_nx, _ny),
      pos((int)(_pos.first * nx), (int)(_pos.second*ny)),
      dir(_dir),
      rules({true, false}),
      colors({WHITE, BLACK}) {}

LangtonAnt::LangtonAnt(int _nx, int _ny, std::pair<float, float> _pos, Direction _dir, std::string _rules, std::vector<Color> _colors)
    : Automaton<uint8_t>(_nx, _ny),
      pos((int)(_pos.first * nx), (int)(_pos.second*ny)),
      dir(_dir),
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

Color LangtonAnt::getColor(uint8_t stateVal) {
    auto color_ind = wrap<uint8_t>(stateVal, colors.size());
    return colors[color_ind];
}
