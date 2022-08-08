// Color constants
#define KOBRA_LOGGER_RESET	"\033[1;0m"
#define KOBRA_LOGGER_RED	"\033[1;31m"
#define KOBRA_LOGGER_GREEN	"\033[1;32m"
#define KOBRA_LOGGER_YELLOW	"\033[1;33m"
#define KOBRA_LOGGER_BLUE	"\033[1;34m"
#define KOBRA_LOGGER_MAGENTA	"\033[1;35m"
#define KOBRA_LOGGER_HEADER	"\033[3;36m"
#define KOBRA_LOGGER_OPTIX	"\033[1;49m"

// Engine headers
#include "../include/logger.hpp"

namespace kobra {

// Static members
std::ostream *Logger::os = &std::cerr;
bool Logger::console = true;
std::mutex Logger::mtx;

// Overloading operator<<
std::ostream &operator<<(std::ostream &os, const log_ok &lt)
        {return os << KOBRA_LOGGER_GREEN;}
        
std::ostream &operator<<(std::ostream &os, const log_warn &lt)
        {return os << KOBRA_LOGGER_YELLOW;}

std::ostream &operator<<(std::ostream &os, const log_error &lt)
        {return os << KOBRA_LOGGER_RED;}

std::ostream &operator<<(std::ostream &os, const log_reset &lt)
        {return os << KOBRA_LOGGER_RESET;}

std::ostream &operator<<(std::ostream &os, const log_notify &lt)
        {return os << KOBRA_LOGGER_BLUE;}

std::ostream &operator<<(std::ostream &os, const log_fatal_error &lt)
        {return os << KOBRA_LOGGER_MAGENTA;}

std::ostream &operator<<(std::ostream &os, const log_header &lt)
        {return os << KOBRA_LOGGER_HEADER;}

std::ostream &operator<<(std::ostream &os, const log_optix &lt)
	{return os << KOBRA_LOGGER_OPTIX;}
	
static std::string time()
{
	auto t = std::time(0);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%H:%M:%S");
	return oss.str();
}

std::ostream &logger(const std::string &source, Log level, const std::string &header, bool source_is_loc)
{
	std::string h = header;

	std::cerr << termcolor::bold;
	if (level == Log::OK) {
		h = "OK";
		std::cerr << termcolor::green;
	} else if (level == Log::ERROR) {
		h = "ERROR";
		std::cerr << termcolor::red;
	} else if (level == Log::WARN) {
		h = "WARN";
		std::cerr << termcolor::yellow;
	} else if (level == Log::INFO) {
		h = "INFO";
		std::cerr << termcolor::blue;
	} else if (level == Log::AUTO) {
		if (h == "OPTIX")
			std::cerr << termcolor::magenta;
		else
			std::cerr << termcolor::blue;
	}
	
	std::cerr << "[" << time() << std::setw(10) << h << "] "
		<< termcolor::reset;

	std::string stripped = source;
	if (!source_is_loc)
		stripped = function_name(stripped);

	std::cerr << termcolor::italic << termcolor::cyan
		<< "[" << stripped << "] " << termcolor::reset;
	return std::cerr;

}

}
