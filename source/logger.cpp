// Engine headers
#include "../include/logger.hpp"

namespace kobra {
	
static std::string time()
{
	auto t = std::time(0);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%H:%M:%S");
	return oss.str();
}

// TODO: some system to avoid repeating the same message
// over and over (CLI and GUI logger)?
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
