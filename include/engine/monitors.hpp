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
        WINDOW *fps_monitor;
};

extern _tui_struct tui;

// Initialize and deinitialize
void init();
void deinit();

}

// Graphical interface
namespace monitors {

}

}

#endif