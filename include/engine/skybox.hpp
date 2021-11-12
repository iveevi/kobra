#ifndef SKYBOX_H_
#define SKYBOX_H_

// Standard headers
#include <string>
#include <vector>

// Engine headers
#include "include/shader.hpp"

namespace mercury {

// Skybox class
class Skybox {
	unsigned int _vao;
	unsigned int _vbo;
	unsigned int _tid;
public:
	Skybox();
	Skybox(const std::vector <std::string> &);

	void draw(Shader &);

	static const float vertices[];
};

}

#endif
