#include "logger.hpp"

// Static members
std::ostream *Logger::os = &std::cerr;
bool Logger::console = true;
std::mutex Logger::mtx;

// Overloading operator<<
std::ostream &operator<<(std::ostream &os, const log_ok &lt)
        {return os << MC_GREEN;}
        
std::ostream &operator<<(std::ostream &os, const log_warn &lt)
        {return os << MC_YELLOW;}

std::ostream &operator<<(std::ostream &os, const log_error &lt)
        {return os << MC_RED;}

std::ostream &operator<<(std::ostream &os, const log_reset &lt)
        {return os << MC_RESET;}

std::ostream &operator<<(std::ostream &os, const log_notify &lt)
        {return os << MC_BLUE;}

std::ostream &operator<<(std::ostream &os, const log_fatal_error &lt)
        {return os << MC_MAGENTA;}