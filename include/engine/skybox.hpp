#ifndef SKYBOX_H_
#define SKYBOX_H_

// Standard headers
#include <string>
#include <vector>

// Engine headers
#include "include/drawable.hpp"
#include "include/shader.hpp"

namespace mercury {

// Skybox class
class Skybox : public Drawable{
	unsigned int _vao;
	unsigned int _vbo;
	unsigned int _tid;
public:
	Skybox();
	Skybox(const std::vector <std::string> &);

	virtual void draw(Shader *) override;

	static const float vertices[];
};

}

#endif
