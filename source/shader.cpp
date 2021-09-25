#include "../include/shader.hpp"

// Standard headers
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

// GLFW
#include "glad/glad.h"
#include <GLFW/glfw3.h>

namespace mercury {

// Helper functions (TODO: different headers?)
std::string read_code(const char *path)
{
	std::ifstream file;
	std::string out;

	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try {
		file.open(path);

		std::stringstream ss;
		ss << file.rdbuf();
		out = ss.str();
	} catch (const std::ifstream::failure &e) {
		// TODO: modify error message:
		// 	include line no propagated from
		// 	include file name
		std::cerr << "Failure loading shader: "
			<< e.what() << std::endl;
	}

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

Shader::Shader(const char *vpath, const char *fpath)
{
	// TODO: separate into functions

	// Reading the code
	std::string vcode_str = read_code(vpath);
	std::string fcode_str = read_code(fpath);

	// Convert to C-strings
	const char *vcode = vcode_str.c_str();
	const char *fcode = fcode_str.c_str();

	// Compiling the shaders
	unsigned int vertex;
	unsigned int fragment;

	int ret;
	char info[512];	// TODO: macro/static const for size

	// vertex shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vcode, NULL);
	glCompileShader(vertex);
	check_program_errors(vertex, "VERTEX");

	// similiar for Fragment Shader TODO in function
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fcode, NULL);
	glCompileShader(fragment);
	check_program_errors(fragment, "FRAGMENT");

	// shader Program
	id = glCreateProgram();
	glAttachShader(id, vertex);
	glAttachShader(id, fragment);
	glLinkProgram(id);
	check_program_errors(id, "PROGRAM");

	// delete the shaders as they're linked into our program now and no longer necessary
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

void Shader::use()
{
	glUseProgram(id);
}

// Setters
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

}
