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
int Shader::_current = -1;
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
static void check_compilation(unsigned int shader, const std::string &type)
{
	static const int ibsize = 1024;
	static const int lsize = 80;
	static const std::string line(lsize, '=');

	GLint success;
	GLchar info[ibsize];

	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (success)
		return;

	glGetShaderInfoLog(shader, ibsize, nullptr, info);

	// Print error messages
	Logger::error() << "Shader compilation error in " << type
		<< " shader (ID = " << shader << ")\n\n";

	std::cout << line << "\n" << info << "\n" << line << "\n\n";
}

static void check_linker(unsigned int shader, const std::string &type)
{
	static const int ibsize = 1024;
	static const int lsize = 80;
	static const std::string line(lsize, '=');

	GLint success;
	GLchar info[ibsize];

	glGetProgramiv(shader, GL_LINK_STATUS, &success);
	if (success)
		return;

	glGetProgramInfoLog(shader, ibsize, nullptr, info);

	// Print error messages
	Logger::error() << "Shader program linking error in " << type
		<< " shader (ID = " << shader << ")\n\n";

	std::cout << line << "\n" << info << "\n" << line << "\n\n";
}

static void check_program_errors(unsigned int shader, const std::string &type)
{
	if (type != "PROGRAM")
		check_compilation(shader, type);
	else
		check_linker(shader, type);
}

Shader::Shader() {}

Shader::Shader(const char *vpath, const char *fpath)
{
	// TODO: separate into functions
	// TODO: use set vertex shader, etc

	// Reading the code
	std::string vcode_str = proc_source(read_code(vpath));
	std::string fcode_str = proc_source(read_code(fpath));

	// Convert to C-strings
	const char *vcode = vcode_str.c_str();
	const char *fcode = fcode_str.c_str();

	// vertex shader
	_vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(_vertex, 1, &vcode, NULL);
	glCompileShader(_vertex);
	check_program_errors(_vertex, "VERTEX");

	// similiar for Fragment Shader TODO in function
	_fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(_fragment, 1, &fcode, NULL);
	glCompileShader(_fragment);
	check_program_errors(_fragment, "FRAGMENT");

	// Shader program
	id = glCreateProgram();
	glAttachShader(id, _vertex);
	glAttachShader(id, _fragment);
	glLinkProgram(id);
	check_program_errors(id, "PROGRAM");

	// Free other programs
	glDeleteShader(_vertex);
	glDeleteShader(_fragment);
}

void Shader::use() const
{
	glUseProgram(id);
	_current = id;
}

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

void Shader::compile()
{
	// Shader program
	id = glCreateProgram();
	glAttachShader(id, _vertex);
	glAttachShader(id, _fragment);
	glLinkProgram(id);
	check_program_errors(id, "PROGRAM");

	// Free other programs
	glDeleteShader(_vertex);
	glDeleteShader(_fragment);
}

// Setters
int get_index(unsigned int id, const std::string &name)
{
	// Cached uniform names
	static std::set <std::string> cached;

	int index = glGetUniformLocation(id, name.c_str());
	if (index < 0 && cached.find(name) == cached.end()) {
		Logger::error() << "No uniform \"" << name
			<< "\" in shader (ID = " << id << ").\n";

		// Add name to the cached list if not present
		cached.insert(cached.begin(), name);
	}
	return index;
}

// Current checker macro
#define CHECK_CURRENT(ftn)						\
	if (id != _current) {						\
		Logger::error("Not using current shader (ID = "		\
			+ std::to_string(_current) + ") for "		\
			+ std::string(ftn));				\
	}

void Shader::set_bool(const std::string &name, bool value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform1i(get_index(id, name), (int) value);
}

void Shader::set_int(const std::string &name, int value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform1i(get_index(id, name), value);
}

void Shader::set_float(const std::string &name, float value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform1f(get_index(id, name), value);
}

void Shader::set_vec2(const std::string &name, const glm::vec2 &value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform2fv(get_index(id, name), 1, &value[0]);
}

void Shader::set_vec2(const std::string &name, float x, float y) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform2f(get_index(id, name), x, y);
}

void Shader::set_vec3(const std::string &name, const glm::vec3 &value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform3fv(get_index(id, name), 1, &value[0]);
}

void Shader::set_vec3(const std::string &name, float x, float y, float z) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform3f(get_index(id, name), x, y, z);
}

void Shader::set_vec4(const std::string &name, const glm::vec4 &value) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform4fv(get_index(id, name), 1, &value[0]);
}

void Shader::set_vec4(const std::string &name,
		float x, float y, float z, float w) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniform4f(get_index(id, name), x, y, z, w);
}

void Shader::set_mat2(const std::string &name, const glm::mat2 &mat) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniformMatrix2fv(get_index(id, name), 1, GL_FALSE, &mat[0][0]);
}

void Shader::set_mat3(const std::string &name, const glm::mat3 &mat) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniformMatrix3fv(get_index(id, name), 1, GL_FALSE, &mat[0][0]);
}

void Shader::set_mat4(const std::string &name, const glm::mat4 &mat) const
{
	CHECK_CURRENT(__PRETTY_FUNCTION__);
	glUniformMatrix4fv(get_index(id, name), 1, GL_FALSE, &mat[0][0]);
}

// Static methods
Shader Shader::from_source(const char *vcode, const char *fcode)
{
	return from_source(std::string(vcode), std::string(fcode));
}

Shader Shader::from_source(const std::string &vcode, const std::string &fcode)
{
	Shader shader;
	shader.set_vertex_shader(vcode);
	shader.set_fragment_shader(fcode);
	shader.compile();
	return shader;
}

}
