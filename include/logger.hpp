#ifndef LOGGER_H_
#define LOGGER_H_

// Standard headers
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>

// Color constants
#define MC_RESET	"\033[0m"
#define MC_RED		"\033[31m"
#define MC_GREEN	"\033[32m"
#define MC_YELLOW	"\033[33m"
#define MC_BLUE		"\033[34m"
#define MC_MAGENTA	"\033[35m"

namespace mercury {

class Logger {
	static std::ostream &_fatal_error() {
		return std::cerr << MC_MAGENTA << "[MERCURY ENGINE: "
			<< time() << ": FATAL ERROR] " << MC_RESET;
	}
public:
	// Aliases
	using tclk = std::chrono::high_resolution_clock;
	using tpoint = std::chrono::high_resolution_clock::time_point;

	static tclk clk;
	static tpoint epoch;

	static void start() {
		epoch = clk.now();
	}

	static std::string time() {
		tpoint tmp = clk.now();

		double sec = std::chrono::duration_cast
			<std::chrono::milliseconds>
			(tmp - epoch).count() / 1000.0;

		int min = sec/60;
		sec = std::fmod(sec, 60);

		std::ostringstream oss;
		oss.precision(3);
		oss << min << "m " << std::fixed << sec << "s";

		return oss.str();
	}

	// As ostream objects
	static std::ostream &ok() {
		return std::cerr << MC_GREEN << "[MERCURY ENGINE: "
			<< time() << "] " << MC_RESET;
	}

	static std::ostream &error() {
		return std::cerr << MC_RED << "[MERCURY ENGINE: "
			<< time() << "] " << MC_RESET;
	}

	static std::ostream &warn() {
		return std::cerr << MC_YELLOW << "[MERCURY ENGINE: "
			<< time() << "] " << MC_RESET;
	}

	// C-string overloads
	static void ok(const char *msg) {
		ok() << msg << std::endl;
	}

	static void error(const char *msg) {
		error() << msg << std::endl;
	}

	static void warn(const char *msg) {
		warn() << msg << std::endl;
	}

	static void fatal_error(const char *msg) {
		_fatal_error() << msg << std::endl;
		exit(-1);
	}

	// C++ string overloads
	static void ok(const std::string &msg) {
		ok(msg.c_str());
	}

	static void error(const std::string &msg) {
		error(msg.c_str());
	}

	static void warn(const std::string &msg) {
		warn(msg.c_str());
	}

	// Cached logging
	static void error_cached(const std::string &msg) {
		static std::set <std::string> cached;

		if (cached.find(msg) == cached.end()) {
			cached.insert(cached.end(), msg);
			error(msg);
		}
	}

	static void warn_cached(const std::string &msg) {
		static std::set <std::string> cached;

		if (cached.find(msg) == cached.end()) {
			cached.insert(cached.end(), msg);
			warn(msg);
		}
	}
};

}

#endif
