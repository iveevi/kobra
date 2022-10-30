#ifndef KOBRA_LOGGER_H_
#define KOBRA_LOGGER_H_

// Standard headers
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>

// Terminal colors header
#include "termcolor/termcolor.hpp"

namespace kobra {

// Helper functions
inline std::string function_name(const std::string &pretty)
{
	// Extract the full function name only
	std::string word;

	for (auto c : pretty) {
		if (c == '(') break;
		else if (c == ' ') word.clear();
		else word += c;
	}

	// If it starts with 'kobra::', remove it
	if (word.substr(0, 7) == "kobra::")
		word = word.substr(7);

	return word;
}

// Macros for logging
enum class Log {OK, ERROR, WARN, INFO, AUTO};

std::ostream &logger(const std::string &, Log level, const std::string & = "", bool = false);

}

// #define KOBRA_LOG_FUNC(type) Logger::type##_from(function_name(__PRETTY_FUNCTION__).c_str())
#define KOBRA_LOG_FUNC(level) kobra::logger(__PRETTY_FUNCTION__, level)

#define LINE_TO_STRING(line) #line
#define LINE_TO_STRING2(line) LINE_TO_STRING(line)

#define KOBRA_LOG_FILE(level) kobra::logger(__FILE__ ": " LINE_TO_STRING2(__LINE__), level, "", true)

#define KOBRA_ASSERT(cond, msg)					\
	if (!(cond)) {						\
		KOBRA_LOG_FUNC(Log::ERROR) << msg << std::endl;	\
		throw std::runtime_error(msg);			\
	}

#endif
