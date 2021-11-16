#ifndef MONITORS_H_
#define MONITORS_H_

// Engine headers
#include "include/logger.hpp"

namespace mercury {

// Text interface
namespace tui {

// Window structures
struct Window {
        WINDOW *main;
        WINDOW *box;
        WINDOW *title;
};

// TUI group struct
struct _tui_struct {
        // Streams
        wbuffer *buf;
        owstream *main;

        // Scrolling variables
        size_t pos = 0;
        bool is_scrolling = false;

        // Windows
        WINDOW *w_title;
        
        WINDOW *w_log;
        WINDOW *w_log_box;
        WINDOW *w_log_title;
        
        WINDOW *w_info;
        WINDOW *w_info_box;
        WINDOW *w_info_title;

        WINDOW *w_controls;
        WINDOW *w_controls_box;
        WINDOW *w_controls_title;

        // Init and deinit
        void init();
        void deinit();

        // Update methods
        void update();          // TODO: should take float delta_t
        void update_logs();
        void update_fps(float);

        // Input
        void input();
};

// Sential object
extern _tui_struct tui;

}

// Graphical interface
namespace monitors {

}

}

#endif