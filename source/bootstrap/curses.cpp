#include <iostream>
#include <ios>
#include <sstream>

// GLFW headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Logger
#include "include/logger.hpp"

// Using declarations
using namespace mercury;

void mouse_callback(GLFWwindow *, double, double) {}

float delta_t = 0.012f;

void ncurses()
{
        initscr();
	noecho();
	curs_set(0);
	scrollok(stdscr, true);
	start_color();

	owstream ows(stdscr);
	ows << "Mercury Ncurses CLI\n";
	refresh();

	Logger::start();
	Logger::switch_stream();
	Logger::ows = &ows;

	int width = 20;
	int height = 10;

	WINDOW *fps_monitor = newwin(height, width, 0, COLS - width);

	do {
		Logger::ok() << "Looped.\n";
		refresh();

		int fps = 1/delta_t;

		box(fps_monitor, 0, 0);
		mvwprintw(fps_monitor, 1, 1, "FPS: %d", fps);
		mvwprintw(fps_monitor, 2, 1, "g - toggle graph");
		wrefresh(fps_monitor);
	} while (getch() != 'q');

	endwin();
}

int main()
{
	ncurses();
}