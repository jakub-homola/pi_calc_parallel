#ifndef PI_CALC_PARALLEL_TIMER_H
#define PI_CALC_PARALLEL_TIMER_H

#include <chrono>

class Timer
{
private:
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point stop_time;
public:
    void start()
    {
        start_time = std::chrono::steady_clock::now();
    }
    void stop()
    {
        stop_time = std::chrono::steady_clock::now();
    }
    double get_seconds()
    {
        std::chrono::duration<double> duration = stop_time - start_time;
        return duration.count();
    }
};

#endif /* PI_CALC_PARALLEL_TIMER_H */
