#include "include/engine/monitors.hpp"

namespace mercury {

namespace tui {

// TUI struct
_tui_struct tui;

// Initialization and deinitialization
void init()
{
        initscr();
	noecho();
	curs_set(0);
	scrollok(stdscr, true);
	start_color();

	// owstream ows(stdscr);
        tui.main = new owstream(stdscr);

	Logger::start();
	Logger::switch_stream();
	Logger::ows = tui.main;

	int width = 20;
	int height = 10;

	tui.fps_monitor = newwin(height, width, 0, COLS - width);
}

void deinit()
{
	// Close windows
	delwin(tui.fps_monitor);
	endwin();
}

}

}