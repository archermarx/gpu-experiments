#include "utils.h"

void delay (uint32_t ms) {
#ifdef WIN_32
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
}