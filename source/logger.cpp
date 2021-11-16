#include "include/logger.hpp"

namespace mercury {

// Static members
std::ostream *Logger::os = &std::cerr;
owstream *Logger::ows = nullptr;
bool Logger::on_main = true;
size_t Logger::line = 1;

// Overloading operator<<
std::ostream &operator<<(std::ostream &os, const log_ok &lt) {return os << MC_GREEN;}
std::ostream &operator<<(std::ostream &os, const log_warn &lt) {return os << MC_YELLOW;}
std::ostream &operator<<(std::ostream &os, const log_error &lt) {return os << MC_RED;}
std::ostream &operator<<(std::ostream &os, const log_reset &lt) {return os << MC_RESET;}
std::ostream &operator<<(std::ostream &os, const log_notify &lt) {return os << MC_BLUE;}
std::ostream &operator<<(std::ostream &os, const log_fatal_error &lt) {return os << MC_MAGENTA;}

}