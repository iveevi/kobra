// Standard headers
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <cstring>
#include <set>

// Engine headers
#include "include/shader.hpp"
#include "include/common.hpp"
#include "include/logger.hpp"

// GLFW
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace mercury {

// Static variables
std::string Shader::_current = "";

std::unordered_map <std::string, std::string> Shader::headers;

// Shader source processing
static void check_header(const std::string &source, std::string &out, int &i)
{
	// Skip white space
	while (i < source.length() && isspace(source[i])) i++;

	// Extract header name
	std::string header;
	while (i < source.length() && source[i] != '\n')
		header += source[i++];

	// Check header database
	if (Shader::headers.find(header) != Shader::headers.end()) {
		out += Shader::headers[header];
	} else {
		// TODO: list all available headers
		Logger::warn() << "No header \"" << header
			<< "\" found. Ignoring include.\n";
	}
}

static void check_directive(const std::string &source, std::string &out, int &i)
{
	// For now, only dealing with #include directive
	std::string preproc;
	for (int j = 0; j < 8 && j + i < source.length(); j++)
		preproc += source[i + j];

	// Advance the index
	i += 7;
	if (preproc == "#include")
		check_header(source, out, ++i);
	else
		out += preproc;
}

static std::string proc_source(const std::string &source)
{
	std::string out;

	// TODO: check available glsl versions, or stick with 3.3
	if (source.find("#version ") == std::string::npos)
		out = "#version 330\n";

	for (int i = 0; i < source.length(); i++) {
		char c = source[i];

		if (c == '#')
			check_directive(source, out, i);
		else
			out += c;
	}

	return out;
}

// TODO: clean
// Source compilation errors
static void check_compilation(unsigned int id, const std::string &type)
{
	static const int ibsize = 1024;
	static const int lsize = 80;
	static const std::string line(lsize, '=');

	GLint success;
	GLchar info[ibsize];

	glGetShaderiv(id, GL_COMPILE_STATUS, &success);
	if (success)
		return;

	glGetShaderInfoLog(id, ibsize, nullptr, info);

	// Print error messages
	Logger::error() << "Shader compilation error in " << type
		<< " shader (id: " << id << ")\n\n";

	std::cout << line << "\n" << info << "\n" << line << "\n\n";
}

static void check_linker(unsigned int id)
{
	static const int ibsize = 1024;
	static const int lsize = 80;
	static const std::string line(lsize, '=');

	GLint success;
	GLchar info[ibsize];

	glGetProgramiv(id, GL_LINK_STATUS, &success);
	if (success)
		return;

	glGetProgramInfoLog(id, ibsize, nullptr, info);

	// Print error messages
	Logger::error() << "Shader program linking error within PROGRAM (id: "
		<< id << ")\n\n";

	std::cout << line << "\n" << info << "\n" << line << "\n\n";
}

static void check_program_errors(unsigned int id, const std::string &type)
{
	if (type == "PROGRAM")
		check_linker(id);
	else
		check_compilation(id, type);
}

Shader::Shader() {}

Shader::Shader(const char *vpath, const char *fpath, const char *gpath)
{
	// Reading the code
	std::string vcode_str = proc_source(read_code(vpath));
	std::string fcode_str = proc_source(read_code(fpath));

	std::string gcode_str;
	if (gpath)
		gcode_str = proc_source(read_code(gpath));

	// Set code and compile
	set_vertex_shader(vcode_str);
	set_fragment_shader(fcode_str);

	if (gpath)
		set_geometry_shader(gcode_str);

	compile();
}

void Shader::use() const
{
	glUseProgram(id);
	glCheckError();
	_current = _name;
}

const std::string &Shader::get_name() const
{
	return _name;
}

void Shader::set_name(const std::string &name)
{
	_name = name;
}

// Source code setters
void Shader::set_vertex_shader(const char *code)
{
	set_vertex_shader(std::string(code));
}

void Shader::set_vertex_shader(const std::string &code)
{
	_vertex = glCreateShader(GL_VERTEX_SHADER);

	// Allocate - TODO: create allocator
	std::string pcode = proc_source(code);
	const char *c_pcode = pcode.c_str();

	// Compiler source
	glShaderSource(_vertex, 1, &c_pcode, NULL);
	glCompileShader(_vertex);
	check_program_errors(_vertex, "VERTEX");
}

void Shader::set_fragment_shader(const char *code)
{
	set_fragment_shader(std::string(code));
}

void Shader::set_fragment_shader(const std::string &code)
{
	_fragment = glCreateShader(GL_FRAGMENT_SHADER);

	// Allocate - TODO: create allocator
	std::string pcode = proc_source(code);
	const char *c_pcode = pcode.c_str();

	// Compiler source
	glShaderSource(_fragment, 1, &c_pcode, NULL);
	glCompileShader(_fragment);
	check_program_errors(_fragment, "FRAGMENT");
}

void Shader::set_geometry_shader(const char *code)
{
	set_geometry_shader(std::string(code));
}

void Shader::set_geometry_shader(const std::string &code)
{
	_geometry = glCreateShader(GL_GEOMETRY_SHADER);
	_has_gshader = true;

	// Preprocess code
	std::string pcode = proc_source(code);
	const char *c_pcode = pcode.c_str();

	// Compiler source
	glShaderSource(_geometry, 1, &c_pcode, NULL);
	glCompileShader(_geometry);
	check_program_errors(_geometry, "GEOMETRY");
}

void Shader::compile()
{
	// Shader program
	id = glCreateProgram();
	
	// Attach shaders
	glAttachShader(id, _vertex);
	glAttachShader(id, _fragment);
	
	if (_has_gshader)
		glAttachShader(id, _geometry);

	// Link and check for errors
	glLinkProgram(id);
	check_program_errors(id, "PROGRAM");

	// Free other programs
	glDeleteShader(_vertex);
	glDeleteShader(_fragment);

	if (_has_gshader)
		glDeleteShader(_geometry);

	// Set name
	_name = "shader_" + std::to_string(id);
}

// Setters
int get_index(const Shader *shader, const std::string &name)
{
	// Cached uniform names
	// TODO: helper function for cached logging?
	//	would avoid looped clutter with certain logging
	//	TODO: replace with error_cached
	static std::set <std::string> cached;

	int index = glGetUniformLocation(shader->id, name.c_str());
	if (index < 0 && cached.find(name) == cached.end()) {
		Logger::error() << "No uniform \"" << name
			<< "\" in shader (id: " << shader->get_name() << ").\n";

		// Add name to the cached list if not present
		cached.insert(cached.begin(), name);
	}
	return index;
}

// Current checker macro
#define CHECK_CURRENT(ftn)						\
	if (_name != _current) {					\
		std::ostringstream oss;					\
		oss << "Not using current shader (id: "			\
			<< _name << ", current: "			\
			<< _current << ") for " << ftn;			\
		Logger::error_cached(oss.str());			\
	}

void Shader::set_bool(const std::string &name, bool value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform1i(get_index(this, name), (int) value);
}

void Shader::set_int(const std::string &name, int value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform1i(get_index(this, name), value);
}

void Shader::set_float(const std::string &name, float value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform1f(get_index(this, name), value);
}

void Shader::set_vec2(const std::string &name, const glm::vec2 &value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform2fv(get_index(this, name), 1, &value[0]);
}

void Shader::set_vec2(const std::string &name, float x, float y) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform2f(get_index(this, name), x, y);
}

void Shader::set_vec3(const std::string &name, const glm::vec3 &value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform3fv(get_index(this, name), 1, &value[0]);
}

void Shader::set_vec3(const std::string &name, float x, float y, float z) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform3f(get_index(this, name), x, y, z);
}

void Shader::set_vec4(const std::string &name, const glm::vec4 &value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform4fv(get_index(this, name), 1, &value[0]);
}

void Shader::set_vec4(const std::string &name,
		float x, float y, float z, float w) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform4f(get_index(this, name), x, y, z, w);
}

void Shader::set_mat2(const std::string &name, const glm::mat2 &mat) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniformMatrix2fv(get_index(this, name), 1, GL_FALSE, &mat[0][0]);
}

void Shader::set_mat3(const std::string &name, const glm::mat3 &mat) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniformMatrix3fv(get_index(this, name), 1, GL_FALSE, &mat[0][0]);
}

void Shader::set_mat4(const std::string &name, const glm::mat4 &mat) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniformMatrix4fv(get_index(this, name), 1, GL_FALSE, &mat[0][0]);
}

// Static methods
Shader Shader::from_source(const char *vcode, const char *fcode, const char *gcode)
{
	return from_source(
		std::string(vcode),
		std::string(fcode),
		(gcode ? std::string(gcode) : "")
	);
}

Shader Shader::from_source(const std::string &vcode, const std::string &fcode, const std::string &gcode)
{
	Shader shader;

	// Set source
	shader.set_vertex_shader(vcode);
	shader.set_fragment_shader(fcode);

	if (!gcode.empty())
		shader.set_geometry_shader(gcode);

	shader.compile();
	return shader;
}

}
