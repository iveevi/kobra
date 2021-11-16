#ifndef LOGGER_H_
#define LOGGER_H_

// Standard headers
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>

// Ncurses
#include <ncurses.h>

// Color constants
#define MC_RESET	"\033[0m"
#define MC_RED		"\033[31m"
#define MC_GREEN	"\033[32m"
#define MC_YELLOW	"\033[33m"
#define MC_BLUE		"\033[34m"
#define MC_MAGENTA	"\033[35m"

namespace mercury {

// Color structs
struct log_ok {};
struct log_warn {};
struct log_error {};
struct log_reset {};
struct log_notify {};
struct log_fatal_error {};

// Overload printing these structs
std::ostream &operator<<(std::ostream &, const log_ok &);
std::ostream &operator<<(std::ostream &, const log_warn &);
std::ostream &operator<<(std::ostream &, const log_error &);
std::ostream &operator<<(std::ostream &, const log_reset &);
std::ostream &operator<<(std::ostream &, const log_notify &);
std::ostream &operator<<(std::ostream &, const log_fatal_error &);

// Window buffer
#define WBUFFER_SIZE	1024

class wbuffer : public std::streambuf {
	WINDOW *			_window = nullptr;
	std::string			_buffer;
public:
	// Buffer structure
	static const size_t buffer_size;
	struct {
		std::string buffer[WBUFFER_SIZE];
		size_t index = 0;
	} buffer;

	// Constructors
	wbuffer() {}
	wbuffer(WINDOW *win) : _window(win) {}

	virtual std::streamsize xsputn(const char *str, std::streamsize size) override {
		// Starting index
		std::streamsize start = 0;

		for (std::streamsize i = 0; i < size; i++) {
			if (str[i] == '\n') {
				_buffer.append(str + start, str + (i + 1));
				sync_ln();
				start = i + 1;
			}
		}

		_buffer.append(str + start, str + size);
		return size;
	}

	virtual int overflow(int c) override {
		_buffer.append(1, c);
		if (c == '\n')
			sync_ln();
		return c;
	}

	void sync_ln() {
		buffer.buffer[buffer.index] = _buffer;
		buffer.index = (buffer.index + 1) % WBUFFER_SIZE;
		sync();
	}

	virtual int sync() override {
		// TODO: check that the _window is not nullptr
		wprintw(_window, "%s", _buffer.c_str());
		_buffer.clear();
		return 0;
	}

	// Friend class
	friend class owstream;
};

class owstream : public std::ostream {
public:
	WINDOW *win = nullptr;

	owstream(wbuffer *buf) : std::ostream(buf) {
		win = buf->_window;
	}
};

// Logger class
class Logger {
	// Static members
	static std::ostream *os;
	static bool on_main;
	static size_t line;

	// For main stream
	static std::ostream &_main_fatal_error() {
		return *os << log_fatal_error() << "[MERCURY ENGINE: "
			<< time() << ": FATAL ERROR] " << log_reset();
	}

	static std::ostream &_main_ok() {
		return *os << log_ok() << "[MERCURY ENGINE: "
			<< time() << "] " << log_reset();
	}

	static std::ostream &_main_error() {
		return *os << log_error() << "[MERCURY ENGINE: "
			<< time() << "] " << log_reset();
	}

	static std::ostream &_main_warn() {
		return *os << log_warn() << "[MERCURY ENGINE: "
			<< time() << "] " << log_reset();
	}

	static std::ostream &_main_notify() {
		return *os << log_notify() << "[MERCURY ENGINE: "
			<< time() << "] " << log_reset();
	}
	
	// For owstream
	enum {
		I_OK = 1,
		I_WARN,
		I_ERROR,
		I_NOTIFY,
		I_FATAL_ERROR,
	};

	static void _print_line() {
		*ows << line++ << ": ";
		ows->flush();
	}

	static void _print_ows_header() {
		*ows << "MERCURY ENGINE: "
			<< time() << " $ ";
		ows->flush();
	}

	static std::ostream &_ows_fatal_error() {
		_print_line();
		wattron(ows->win, COLOR_PAIR(I_FATAL_ERROR));
		_print_ows_header();
		wattroff(ows->win, COLOR_PAIR(I_FATAL_ERROR));
		
		return *ows;
	}

	static std::ostream &_ows_ok() {
		_print_line();
		wattron(ows->win, COLOR_PAIR(I_OK));
		_print_ows_header();
		wattroff(ows->win, COLOR_PAIR(I_OK));
		
		return *ows;
	}

	static std::ostream &_ows_error() {
		_print_line();
		wattron(ows->win, COLOR_PAIR(I_ERROR));
		_print_ows_header();
		wattroff(ows->win, COLOR_PAIR(I_ERROR));
		
		return *ows;
	}

	static std::ostream &_ows_warn() {
		_print_line();
		wattron(ows->win, COLOR_PAIR(I_WARN));
		_print_ows_header();
		wattroff(ows->win, COLOR_PAIR(I_WARN));
		
		return *ows;
	}

	static std::ostream &_ows_notify() {
		_print_line();
		wattron(ows->win, COLOR_PAIR(I_NOTIFY));
		_print_ows_header();
		wattroff(ows->win, COLOR_PAIR(I_NOTIFY));
		
		return *ows;
	}
public:
	// Aliases
	using tclk = std::chrono::high_resolution_clock;
	using tpoint = std::chrono::high_resolution_clock::time_point;

	static tclk clk;
	static tpoint epoch;

	static void start() {
		epoch = clk.now();

		// TODO: setup all color indexes
		init_pair(I_OK, COLOR_GREEN, COLOR_BLACK);
		init_pair(I_WARN, COLOR_YELLOW, COLOR_BLACK);
		init_pair(I_ERROR, COLOR_RED, COLOR_BLACK);
		init_pair(I_NOTIFY, COLOR_BLUE, COLOR_BLACK);
		init_pair(I_FATAL_ERROR, COLOR_MAGENTA, COLOR_BLACK);
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
	
	// For main stream
	static std::ostream &ok() {
		if (on_main)
			return _main_ok();

		return _ows_ok();
	}

	static std::ostream &error() {
		if (on_main)
			return _main_error();

		return _ows_error();
	}

	static std::ostream &warn() {
		if (on_main)
			return _main_warn();

		return _ows_warn();
	}

	static std::ostream &notify() {
		if (on_main)
			return _main_notify();

		return _ows_notify();
	}

	static void switch_stream() {
		on_main = !on_main;
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
		if (on_main) {
			_main_fatal_error() << msg << std::endl;
		} else {
			_ows_fatal_error() << msg << std::endl;
			_ows_fatal_error() << "Press any key to exit." << std::endl;
			getch();
			endwin();
		}

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
	
	static owstream *ows;
};

}

#endif
