#include "automaton.h"
#include "utils.h"

void Automaton::draw(Canvas& canvas) {
    int pixelIndex = 0;
    for (auto& row: state) {
        for (bool stateVal: row) {
            auto color = getColor(stateVal);
            canvas.contents[pixelIndex] = color.r;
            canvas.contents[pixelIndex+1] = color.g;
            canvas.contents[pixelIndex+2] = color.b;
            canvas.contents[pixelIndex+3] = 255;
            pixelIndex += 4;
        }
    }
}
