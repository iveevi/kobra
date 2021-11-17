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

namespace monitors {

/* void fps_monitor_initializer()
{
	// TODO: display fps counter here

	// Uncap FPS
	glfwSwapInterval(0);

	// For text rendering
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Load font
	winman.load_font(1);

	// Set line width
	glLineWidth(5.0f);

	// Fill vertices with 0
	for (size_t i = 0; i < 3 * 10; i++)
		fps_vertices[i] = 0;

	// Create axes
	axes = SVA3({
		0,	100,	0,
		0,	0,	0,
		100,	0,	0,
	});

	// Allocate graph buffer
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glBindVertexArray(vao);

	generate_arrays();

	// Create and configure base graphing shader
	basic = Shader::from_source(basic_vert, basic_frag);

	basic.use();
	//basic.set_vec3("ecolor", {1.0, 1.0, 1.0});
	basic.set_mat4("projection", glm::ortho(-10.0f, 110.0f, -10.0f, 110.0f));

	// Set text
	text_fps = ui::Text("FPS", 100, 10, 0.9, {1.0, 0.5, 1.0});

	// TODO: move to init or smthing
}

void fps_monitor_renderer()
{
	// Static timer
	static float time = 0.0f;
	static float totfps = 0.0f;
	static float iters = 0.0f;
	static float avgfps = 0.0f;

	// Clear the color
	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// Get time stuff
	float current_frame = glfwGetTime();
	delta_time = current_frame - last_frame;
	last_frame = current_frame;
	time += delta_time;

	int fps = 1/delta_time;
	totfps += fps;
	iters++;

	if (time > 0.5f) {
		// Delete the first point
		fps_vertices.erase(fps_vertices.begin(), fps_vertices.begin() + 3);

		// 0 -> 600 fps
		avgfps = totfps/iters;

		float normalized = 100.0f * avgfps/600.0f;
		fps_vertices.push_back(100.0f);
		fps_vertices.push_back(normalized);
		fps_vertices.push_back(0.0f);

		// Shift other points

		// TODO: += fields (getter)
		for (size_t i = 0; i < fps_vertices.size(); i += 3) {
			float px = fps_vertices[i];
			fps_vertices[i] = std::max(px - 10.0f, 0.0f);
		}

		// Regenerate buffer data
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glCheckError();

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * fps_vertices.size(),
			&fps_vertices[0], GL_STATIC_DRAW);

		// Reset time
		time = 0.0f;
	}

	// Draw the graph
	basic.use();
	basic.set_vec3("ecolor", {1.0, 1.0, 1.0});

	glBindVertexArray(vao);
	glCheckError();

	glDrawArrays(GL_LINE_STRIP, 0, fps_vertices.size()/3);
	glCheckError();

	// Draw axes
	basic.set_vec3("ecolor", {0.6, 0.6, 0.6});

	axes.draw(GL_LINE_STRIP);

	// Draw text
	text_fps.set_str(std::to_string(delta_time).substr(0, 6)
		+ "s delta, " + std::to_string(avgfps).substr(0, 6) + " fps");
	text_fps.draw(*winman.cres.text_shader);
} */

}

}