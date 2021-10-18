#include "include/init.hpp"
#include "include/logger.hpp"
#include "include/common.hpp"
#include "include/ui/text.hpp"
#include "include/ui/ui_element.hpp"

// Extra GLM headers
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Make sure that relative path is defined
#ifndef MERCURY_SOURCE_DIR

#error MERCURY_SOURCE_DIR not defined

#endif

// Defined constants
#define DEFAULT_WINDOW_WIDTH	800.0
#define DEFAULT_WINDOW_HEIGHT	600.0

namespace mercury {

// Initial window dimensions
Window cwin;

// TODO: this is no longer necessary
glm::vec2 transform(const glm::vec2 &in)
{
	// TODO: this should not be using the global cwin
	return glm::vec2 {
		(in.x - cwin.width/2)/(cwin.width/2),
		-(in.y - cwin.height/2)/(cwin.height/2)
	};
}

// Character static variables
Shader Char::shader;

// Character map
std::unordered_map <char, Char> cmap;

// UI Element static variables
float ui::UIElement::swidth = -1;
float ui::UIElement::sheight = -1;
Shader ui::UIElement::shader;
glm::mat4 ui::UIElement::projection;

// Text static variables
float ui::Text::swidth = -1;
float ui::Text::sheight = -1;

// Logger static variables
Logger::tclk Logger::clk;
Logger::tpoint Logger::epoch;

// Reassign orthos
void update_screen_size(float width, float height)
{
	// Logger::ok(std::string("width = ") + std::to_string(width) + ", height = " + std::to_string(height));
	// TODO: use gl get dims or smthing?
	cwin.width = width;
	cwin.height = height;

	// TODO: is this correct/needed?
	ui::UIElement::set_projection(0.0f, width, 0.0f, height, width, height);
}

// Window size change callback
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	Logger::ok(std::string("width = ") + std::to_string(width) + ", height = " + std::to_string(height));
	glViewport(0, 0, width, height);
	Logger::ok("again: " + std::string("width = ") + std::to_string(width) + ", height = " + std::to_string(height));
	update_screen_size(width, height);

#ifdef MERCURY_DEBUG

	// TODO: update all orthos
	Logger::ok() << "Resized window to " << width
		<< " x " << height << std::endl;

#endif

}

// Mouse interrupt callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	// TODO: send change in mouse position anyways
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		// TODO: put into function in init.hpp
		double xpos, ypos;

		glfwGetCursorPos(window, &xpos, &ypos);
		mercury::cwin.mouse_handler.publish(
			{(mercury::MouseBus::Type) action, {xpos, ypos}}
		);
	}
}

// Private static methods
static void init_glfw()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__

	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	cwin.window = glfwCreateWindow(DEFAULT_WINDOW_WIDTH,
		DEFAULT_WINDOW_HEIGHT, "Mercury", nullptr, nullptr);
	if (cwin.window == nullptr) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(cwin.window);
	glfwSetFramebufferSizeCallback(cwin.window, framebuffer_size_callback);
	glfwSetMouseButtonCallback(cwin.window, mouse_button_callback);

	// Now initialize glad
	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(-1);
	}

#ifndef MERCURY_DEBUG

	glEnable(GL_CULL_FACE);

#endif

	// For text rendering
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void load_fonts()
{
	// Load default font
	FT_Library ft;
	if (FT_Init_FreeType(&ft)) {
		std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
		exit(-1);
	}

	FT_Face face;
	int ret;
	if ((ret = FT_New_Face(ft, MERCURY_SOURCE_DIR "/resources/fonts/arial.ttf", 0, &face))) {
		std::cout << "ERROR::FREETYPE: Failed to load font (" << ret << ")" << std::endl;
		exit(-1);
	}

	FT_Set_Pixel_Sizes(face, 0, 48);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // disable byte-alignment restriction
	for (unsigned char c = 0; c < 128; c++) {
		// load character glyph
		if (FT_Load_Char(face, c, FT_LOAD_RENDER))
		{
			std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
			continue;
		}
		// generate texture
		unsigned int texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(
				GL_TEXTURE_2D,
				0,
				GL_RED,
				face->glyph->bitmap.width,
				face->glyph->bitmap.rows,
				0,
				GL_RED,
				GL_UNSIGNED_BYTE,
				face->glyph->bitmap.buffer
		);
		// set texture options
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// now store character for later use
		Char character = {
			texture,
			glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
			glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
			(unsigned int) face->glyph->advance.x
		};

		cmap.insert({c, character});
	}

	// Release resources
	FT_Done_Face(face);
	FT_Done_FreeType(ft);

	// Create the text shader
	Char::shader = Shader(
		MERCURY_SOURCE_DIR "/resources/shaders/font_shader.vs",
		MERCURY_SOURCE_DIR "/resources/shaders/font_shader.fs"
	);

	// Set default projection
	ui::Text::set_projection(800, 600);
}

void init()
{
	// Start the logger
	Logger::start();

	// TODO: need to check error codes
	Logger::ok("Started Mercury Engine.");

	// Very first thing
	init_glfw();

	// TODO: need to check error codes
	Logger::ok("Successfully initialized GLFW.");

	load_fonts();

	// TODO: need to check error codes
	Logger::ok("Successfully loaded all fonts.");

	// Create the default shader
	ui::UIElement::shader = Shader(
		MERCURY_SOURCE_DIR "/resources/shaders/shape_shader.vs",
		MERCURY_SOURCE_DIR "/resources/shaders/shape_shader.fs"
	);
	
	// Set default projection
	ui::UIElement::set_projection(0, 800, 0, 600, 800, 600);

	// Set initial screen parameters
	update_screen_size(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
}

}
