#include "Timing.h"

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
