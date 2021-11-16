// Engine headers
#include "include/init.hpp"
#include "include/engine/monitors.hpp"

namespace mercury {

namespace tui {

// TUI struct
_tui_struct tui;

// TODO: add a window class (with title box as well...)

// Initialization and deinitialization
void _tui_struct::init()
{
        static const std::string title = "MERCURY ENGINE";
        static const size_t pos = (80 - title.length())/2;

        static const std::string log_title = "LOGS";
        static const size_t log_pos = (80 - log_title.length())/2;
        
        static const std::string info_title = "ENGINE INFO";
        static const size_t info_pos = (40 - info_title.length())/2;

        static const std::string controls_title = "CONTROLS";
        static const size_t controls_pos = (40 - controls_title.length())/2;

        // Ncurses initialization
        initscr();
	noecho();
	curs_set(0);
	start_color();

        // Create windows
        w_title = newwin(3, 82, 0, 0);
        
        w_log = newpad(1024, 80); // newwin(17, 80, 7, 1);
        keypad(w_log, true);
	scrollok(w_log, true);
        nodelay(w_log, true);
 
        w_log_box = newwin(22, 82, 3, 0);
        w_log_title = newwin(3, 80, 4, 1);

	w_info = newwin(5, 38, 4, 84);
	w_info_box = newwin(10, 40, 0, 83);
        w_info_title = newwin(3, 38, 1, 84);

        w_controls = newwin(5, 38, 14, 84);
        w_controls_box = newwin(10, 40, 10, 83);
        w_controls_title = newwin(3, 38, 11, 84);
        
        // Draw the boxes
        box(w_log_box, 0, 0);
        wrefresh(w_log_box);
        
        box(w_info_box, 0, 0);
        wrefresh(w_info_box);

        box(w_controls_box, 0, 0);
        wrefresh(w_controls_box);

        // Write the titles
        box(w_title, 0, 0);
	mvwprintw(w_title, 1, pos, title.c_str());
        wrefresh(w_title);

        box(w_log_title, 0, 0);
        mvwprintw(w_log_title, 1, log_pos, log_title.c_str());
        wrefresh(w_log_title);

        box(w_info_title, 0, 0);
        mvwprintw(w_info_title, 1, info_pos, info_title.c_str());
        wrefresh(w_info_title);

        box(w_controls_title, 0, 0);
        mvwprintw(w_controls_title, 1, controls_pos, controls_title.c_str());
        wrefresh(w_controls_title);
        
        // Configure the logger
        buf = new wbuffer(w_log);
        main = new owstream(buf);

	Logger::start();
	Logger::switch_stream();
	Logger::ows = main;
}

void _tui_struct::deinit()
{
	// Close windows
        delwin(w_title);
        
        delwin(w_log);
        delwin(w_log_box);
        delwin(w_log_title);
	
        delwin(w_info);
        
        delwin(w_controls);

	endwin();
}

// Updating
void _tui_struct::update() // TODO: make more consistent naming: update(), update_logs()
{
        update_logs();

        // Write the controls
        box(w_controls_box, 0, 0);
        mvwprintw(w_controls, 0, 0, "q - quit");
        mvwprintw(w_controls, 1, 0, "p - pause/unpause logger");
        mvwprintw(w_controls, 2, 0, "g - toggle graph");
        mvwprintw(w_controls, 3, 0, "up - scroll up logs");
        mvwprintw(w_controls, 4, 0, "down - scroll down logs");
        wrefresh(w_controls);
}

void _tui_struct::update_logs()
{
        // Refresh the logs
        // wrefresh(w_log);
        prefresh(w_log, 100, 0, 7, 1, 23, 79);
}

void _tui_struct::update_fps(float delta_t)
{
        // Rolling buffer structure
        static const size_t buffer_size = 50;
        static struct {
                float buffer[buffer_size];
                size_t pos = 0;
                float avg = 0.0f;
        } fps_buffer;

        // Calculate the fps
        if (delta_t <= 0.0f)    // Quit if delta_t is null
                return;
        
	float fps = 1/delta_t;

        // Calculate average and insert into buffer
        fps_buffer.buffer[fps_buffer.pos] = fps;
        fps_buffer.pos = (fps_buffer.pos + 1) % buffer_size;

        // Calculate average every full buffer
        if (fps_buffer.pos == 0) {
                fps_buffer.avg = 0.0f;
                for (size_t i = 0; i < buffer_size; i++)
                        fps_buffer.avg += fps_buffer.buffer[i];
                fps_buffer.avg /= buffer_size;
        }

        // Display the information
        int avg_fps = (int) fps_buffer.avg;

	box(w_info_box, 0, 0);
	mvwprintw(w_info, 0, 0, "Average Frame Per Second: %d", avg_fps);
        mvwprintw(w_info, 1, 0, "Average Delta Time:       %f", 1/fps_buffer.avg);
        mvwprintw(w_info, 2, 0, "Time Since Engine Start:  %s", Logger::time().c_str());
	wrefresh(w_info);
}

// Input
void _tui_struct::input()
{
        // Get input
        int c = wgetch(w_log);

        // Handle input
        switch (c) {
        case ERR:
                break;
        case 'q':
                deinit();
                winman.close_all();
                break;
        case KEY_DOWN:
                wscrl(w_log, 1);
                break;
        case KEY_UP:
                wscrl(w_log, -1);
                break;
        }
}

}

}