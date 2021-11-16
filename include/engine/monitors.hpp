#ifndef MONITORS_H_
#define MONITORS_H_

// Engine headers
#include "include/logger.hpp"

namespace mercury {

// Text interface
namespace tui {

// TUI group struct
struct _tui_struct {
        // Streams
        owstream *main;

        // Windows
        WINDOW *w_log;
        WINDOW *w_log_box;
        WINDOW *w_title;
        WINDOW *w_fps;

        // Init and deinit
        void init();
        void deinit();

        // Update methods
        void update_logs();
        void update_fps(float);
};

// Sential object
extern _tui_struct tui;

}

// Graphical interface
namespace monitors {

}

}

#endif