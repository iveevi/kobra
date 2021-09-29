#include "../include/shader.hpp"

// Standard headers
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <utility>

// GLFW
#include "../glad/glad.h"
#include <GLFW/glfw3.h>

namespace mercury {

void error_file(const char *path)
{
	std::cerr << "Could not load file \"" << path << "\"\n";
	exit(-1);

	// TODO: should throw a retrievable exception later
	// a vector of potential files...?
}

// Helper functions (TODO: different headers?)
std::string read_code(const char *path)
{
	std::ifstream file;
	std::string out;

	file.open(path);
	if (!file)
		error_file(path);

	std::stringstream ss;
	ss << file.rdbuf();
	out = ss.str();

	return out;
}

// TODO: clean
void check_program_errors(unsigned int shader, const std::string &type)
{
	GLint success;
	GLchar infoLog[1024];
	if(type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if(!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if(!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

Shader::Shader() {}

Shader::Shader(const char *vpath, const char *fpath)
{
	// TODO: separate into functions

	// Reading the code
	std::string vcode_str = read_code(vpath);
	std::string fcode_str = read_code(fpath);

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
}

void Shader::set_vertex_shader(const char *code)
{
	_vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(_vertex, 1, &code, NULL);
	glCompileShader(_vertex);
	check_program_errors(_vertex, "VERTEX");
}

void Shader::set_fragment_shader(const char *code)
{
	_fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(_fragment, 1, &code, NULL);
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
// TODO: erroor handling for these attributes
// TODO: print a possbility of not doing .use() before
void Shader::set_bool(const std::string &name, bool value) const
{
	glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
}

void Shader::set_int(const std::string &name, int value) const
{
	glUniform1i(glGetUniformLocation(id, name.c_str()), value);
}

void Shader::set_float(const std::string &name, float value) const
{
	glUniform1f(glGetUniformLocation(id, name.c_str()), value);
}

void Shader::set_vec2(const std::string &name, const glm::vec2 &value) const
{
	glUniform2fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
}

void Shader::set_vec2(const std::string &name, float x, float y) const
{
	glUniform2f(glGetUniformLocation(id, name.c_str()), x, y);
}

void Shader::set_vec3(const std::string &name, const glm::vec3 &value) const
{
	glUniform3fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
}

void Shader::set_vec3(const std::string &name, float x, float y, float z) const
{
	glUniform3f(glGetUniformLocation(id, name.c_str()), x, y, z);
}

void Shader::set_vec4(const std::string &name, const glm::vec4 &value) const
{
	glUniform4fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
}

void Shader::set_vec4(const std::string &name,
		float x, float y, float z, float w) const
{
	glUniform4f(glGetUniformLocation(id, name.c_str()), x, y, z, w);
}

void Shader::set_mat2(const std::string &name, const glm::mat2 &mat) const
{
	glUniformMatrix2fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE,
			&mat[0][0]);
}

void Shader::set_mat3(const std::string &name, const glm::mat3 &mat) const
{
	glUniformMatrix3fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE,
			&mat[0][0]);
}

void Shader::set_mat4(const std::string &name, const glm::mat4 &mat) const
{
	glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE,
			&mat[0][0]);
}

// Static methods
Shader Shader::from_source(const char *vcode, const char *fcode)
{
	Shader shader;
	shader.set_vertex_shader(vcode);
	shader.set_fragment_shader(fcode);
	shader.compile();
	return shader;
}

}
