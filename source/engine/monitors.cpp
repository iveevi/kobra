#include "include/engine/monitors.hpp"

namespace mercury {

namespace tui {

// TUI struct
_tui_struct tui;

// Initialization and deinitialization
void _tui_struct::init()
{
        initscr();
	noecho();
	curs_set(0);
	start_color();

        // Create windows
        w_log = newwin(20, 80, 4, 1);
	scrollok(w_log, true);
 
        w_log_box = newwin(22, 82, 3, 0);

        w_title = newwin(3, 82, 0, 0);
	w_fps = newwin(10, 20, 0, COLS - 20);
        
        // Draw the log box
        box(w_log_box, 0, 0);
        wrefresh(w_log_box);
        
        // Configure the logger
        main = new owstream(w_log);

	Logger::start();
	Logger::switch_stream();
	Logger::ows = main;
}

void _tui_struct::deinit()
{
	// Close windows
        delwin(w_log);
        delwin(w_log_box);
        delwin(w_title);
	delwin(w_fps);
	endwin();
}

// Updating
void _tui_struct::update_logs()
{
        static const std::string title = "Mercury Engine";
        static const size_t pos = (80 - title.length())/2;
	
        // Refresh the logs
        wrefresh(w_log);

        // Write the title
        box(w_title, 0, 0);
	mvwprintw(w_title, 1, pos, title.c_str());
        wrefresh(w_title);
}

void _tui_struct::update_fps(float delta_t)
{
	int fps = 1/delta_t;

	box(w_fps, 0, 0);
	mvwprintw(w_fps, 1, 1, "FPS: %d", fps);
	mvwprintw(w_fps, 2, 1, "g - toggle graph");
	wrefresh(w_fps);
}

}

}