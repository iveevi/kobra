#ifndef KOBRA_LOGGER_H_
#define KOBRA_LOGGER_H_

// Standard headers
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
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

// Logging handlers
using LogHandler = std::function <
	void (Log, const std::string &, const std::string &,
		const std::string &, const std::string &)
>;

namespace detail {

// Global log handlers
extern std::map <void *, LogHandler> log_handlers;

// Custom streambuf for piping logging
class LogStreamBuf : public std::basic_streambuf <char> {
	Log level;
	std::string time;
	std::string header;
	std::string source;
	std::string line;
public:
	LogStreamBuf(std::streambuf *sb) : main_stream(sb) {}

	void current_context(Log l, const std::string &t, const std::string &h, const std::string &s) {
		for (auto &h : log_handlers)
			h.second(l, time, header, source, line);

		time = t;
		header = h;
		source = s;
		level = l;

		line.clear();
	}
protected:
	std::streambuf *main_stream;

	int_type overflow(int_type c) override {
		if (c == traits_type::eof()) {
			return traits_type::not_eof(c);
		} else {
			// Convert char to string
			line += c;

			// Send to main stream
			main_stream->sputc(c);

			return c;
		}
	}

	std::streamsize xsputn(const char *s, std::streamsize n) override {
		// Convert char to string
		std::string str(s, n);
		line += str;

		// Send to main stream
		main_stream->sputn(s, n);

		return n;
	}
};

// Custom stream for piping logging
class LogStream : public std::basic_ostream <char> {
public:
	LogStream(std::ostream &os)
			: std::basic_ostream <char>
			(new LogStreamBuf(os.rdbuf())) {}

	void current_context(Log l, const std::string &t, const std::string &h, const std::string &s) {
		static_cast <LogStreamBuf *>
			(rdbuf())->current_context(l, t, h, s);
	}
};

}

// Add a log handler
inline void add_log_handler(void *user, LogHandler handler)
{
	detail::log_handlers[user] = handler;
}

// Remove a log handler
inline void remove_log_handler(void *user)
{
	detail::log_handlers.erase(user);
}

// Logging function
detail::LogStream &logger(const std::string &, Log level, const std::string & = "", bool = false);

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
