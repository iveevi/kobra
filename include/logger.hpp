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
#include "termcolor/include/termcolor/termcolor.hpp"

namespace kobra {

// Color structs
struct log_ok {};
struct log_warn {};
struct log_error {};
struct log_reset {};
struct log_notify {};
struct log_fatal_error {};
struct log_header {};

// Custom headers
struct log_optix {
	static constexpr const char name[] = "OPTIX";
};

// Overload printing these structs
std::ostream &operator<<(std::ostream &, const log_ok &);
std::ostream &operator<<(std::ostream &, const log_warn &);
std::ostream &operator<<(std::ostream &, const log_error &);
std::ostream &operator<<(std::ostream &, const log_reset &);
std::ostream &operator<<(std::ostream &, const log_notify &);
std::ostream &operator<<(std::ostream &, const log_fatal_error &);
std::ostream &operator<<(std::ostream &, const log_header &);
std::ostream &operator<<(std::ostream &, const log_optix &);

// Logger class
// TODO: completely put in source file
class Logger {
	// Static members
	static std::ostream *	os;
	static bool		console; // Whether the stream is console (for colors)
	static std::mutex	mtx;

	// For main stream
	// TODO: engine callbacks to these functions
	static constexpr int _header_width = 20;

	static std::ostream &_main_fatal_error() {
		mtx.lock();
		if (console) *os << log_fatal_error();
		*os << "[" << time() << std::setw(_header_width)
			<< "FATAL ERROR] ";
		if (console) *os << log_reset();
		if (!console) os->flush();
		mtx.unlock();
		return *os;
	}

	static std::ostream &_main_plain() {
		if (!console) os->flush();
		return *os;
	}

	static std::ostream &_main_ok() {
		mtx.lock();
		if (console) *os << log_ok();
		*os << "[" << time() << std::setw(_header_width)
			<< "OK] ";
		if (console) *os << log_reset();
		if (!console) os->flush();
		mtx.unlock();
		return *os;
	}

	static std::ostream &_main_error() {
		mtx.lock();
		if (console) *os << log_error();
		*os << "[" << time() << std::setw(_header_width)
			<< "ERROR] ";
		if (console) *os << log_reset();
		if (!console) os->flush();
		mtx.unlock();
		return *os;
	}

	static std::ostream &_main_warn() {
		mtx.lock();
		if (console) *os << log_warn();
		*os << "[" << time() << std::setw(_header_width)
			<< "WARNING] ";
		if (console) *os << log_reset();
		if (!console) os->flush();
		mtx.unlock();
		return *os;
	}

	// TODO: refactor to info
	static std::ostream &_main_notify() {
		mtx.lock();
		if (console) *os << log_notify();
		*os << "[" << time() << std::setw(_header_width)
			<< "INFO] ";
		if (console) *os << log_reset();
		if (!console) os->flush();
		mtx.unlock();
		return *os;
	}

	// Custom headers
	template <class Header>
	static std::ostream &_main_custom() {
		mtx.lock();
		if (console) *os << log_optix();
		*os << "[" << time() << std::setw(1)
			<< Header::name << "] ";
		if (console) *os << log_reset();
		if (!console) os->flush();
		mtx.unlock();
		return *os;
	}
public:
	static std::string time() {
		auto t = std::time(0);
		auto tm = *std::localtime(&t);
		std::ostringstream oss;
		oss << std::put_time(&tm, "%H:%M:%S");
		return oss.str();
	}

	// For main stream
	static std::ostream &plain() {
		return _main_plain();
	}

	static std::ostream &ok() {
		return _main_ok();
	}

	static std::ostream &error() {
		return _main_error();
	}

	static std::ostream &warn() {
		return _main_warn();
	}

	static std::ostream &notify() {
		return _main_notify();
	}

	template <class Header>
	static std::ostream &custom() {
		return _main_custom <Header> ();
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

	static void notify(const char *msg) {
		notify() << msg << std::endl;
	}

	static void fatal_error(const char *msg) {
		_main_fatal_error() << msg << std::endl;
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

	static void notify(const std::string &msg) {
		notify(msg.c_str());
	}

	// With bracketed origin
	// TODO: function to write the origined message
	static void ok(const char *origin, const char *msg) {
		ok() << log_header() << "[" << origin << "] "
			<< log_reset() << msg << std::endl;
	}

	static void error(const char *origin, const char *msg) {
		error() << log_header() << "[" << origin << "] "
			<< log_reset() << msg << std::endl;
	}

	static void warn(const char *origin, const char *msg) {
		warn() << log_header() << "[" << origin << "] "
			<< log_reset() << msg << std::endl;
	}

	static void notify(const char *origin, const char *msg) {
		notify() << log_header() << "[" << origin << "] "
			<< log_reset() << msg << std::endl;
	}

	// Bracketed origin and stream
	static std::ostream &ok_from(const char *origin) {
		return ok() << log_header() << "[" << origin
			<< "] " << log_reset();
	}

	static std::ostream &error_from(const char *origin) {
		return error() << log_header() << "[" << origin
			<< "] "	<< log_reset();
	}

	static std::ostream &warn_from(const char *origin) {
		return warn() << log_header() << "[" << origin
			<< "] " << log_reset();
	}

	static std::ostream &notify_from(const char *origin) {
		return notify() << log_header() << "[" << origin
			<< "] " << log_reset();
	}

	template <class Header>
	static std::ostream &custom_from(const char *origin) {
		return custom <Header> () << log_header() << "[" << origin
			<< "] " << log_reset();
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

	// Switching streams
	static void switch_console(std::ostream &new_os) {
		// Delete the old stream if not console
		if (!console) delete os;

		// Assumes that new_os is a standard IO stream
		os = &new_os;
		console = true;
	}

	static void switch_file(const std::string &file) {
		// Delete the old stream if not console
		if (!console) delete os;

		// Create ofstream at file
		os = new std::ofstream(file);
		console = false;
	}
};

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
// TODO: option to swthich function and pretty function
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
