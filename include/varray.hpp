#ifndef VARRAY_H_
#define VARRAY_H_

// GLFW headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Engine headers
#include "include/common.hpp"

namespace mercury {

// TODO: add a source file as well

// Vertex buffer struct
//	fields is the number of floating point
//	numbers that represent a single vertex
//
//	for plain 3d vertices, this would be 3
//	if normals are included, it would become 6
template <unsigned int fields>
struct VertexArray {
	unsigned int vao;
	unsigned int vbo;

	// Number of vertices
	size_t size;

	// TODO: what about index buffer?
	virtual void draw(GLenum mode) const {
		glBindVertexArray(vao);
		glDrawArrays(mode, 0, size);
		glCheckError();
	}
};

// Vertex array that cannot be modified
template <unsigned int fields>
struct StaticVA : public VertexArray <fields> {
	StaticVA() {}

	StaticVA(const std::vector <float> &verts) {
		// Set size
		this->size = verts.size() / fields;

		// Allocate buffers
		glGenVertexArrays(1, &this->vao);
		glGenBuffers(1, &this->vbo);
		glBindVertexArray(this->vao);

		// Load buffer
		glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * verts.size(),
			&verts[0], GL_STATIC_DRAW);

		glVertexAttribPointer(0, fields, GL_FLOAT, GL_FALSE,
			fields * sizeof(float), (void *) 0);
		glEnableVertexAttribArray(0);
		glCheckError();
	}
};

}

#endif