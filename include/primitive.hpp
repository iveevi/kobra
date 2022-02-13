#ifndef PRIMITIVE_H_
#define PRIMITIVE_H_

// Engine headers
#include "core.hpp"
#include "material.hpp"
#include "transform.hpp"
#include "types.h"

// Primitive structures
struct Primitive {
	float		id = OBJECT_TYPE_NONE;
	Transform	transform;
	Material	material;

	// Primitive constructors
	Primitive() {}
	Primitive(float x, const Transform &t, const Material &m)
			: id(x), transform(t), material(m) {}

	// Virtual object destructor
	virtual ~Primitive() {}

	// Write data to aligned_vec4 buffer (inherited)
	virtual void write(Buffer &buffer) const = 0;

	// Write full object data
	// TODO: pass paramters as a struct
	virtual void write_to_buffer(Buffer &buffer, Buffer &materials, Indices &indices) {
		// Deal with material
		uint mati = materials.size();
		material.write_to_buffer(materials);
		float index = *reinterpret_cast <float *> (&mati);

		// Push ID and material, then everything else
		buffer.push_back(aligned_vec4 {
			glm::vec4(id, index, 0.0, 0.0)
		});

		this->write(buffer);
	}

	// Write full object data, but takes index to material
	virtual void write_to_buffer(Buffer &buffer, Indices &indices, uint mati) {
		// Deal with material
		float index = *reinterpret_cast <float *> (&mati);

		// Push ID and material, then everything else
		buffer.push_back(aligned_vec4 {
			glm::vec4(id, index, 0.0, 0.0)
		});

		this->write(buffer);
	}
};

// Triangle primitive
struct Triangle : public Primitive {
	glm::vec3 a;
	glm::vec3 b;
	glm::vec3 c;

	Triangle() {}
	Triangle(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c,
			const Material &m)
			: Primitive(OBJECT_TYPE_TRIANGLE, Transform(), m),
			a(a), b(b), c(c) {}

	void write(Buffer &buffer) const override {
		buffer.push_back(aligned_vec4(a));
		buffer.push_back(aligned_vec4(b));
		buffer.push_back(aligned_vec4(c));
	}
};

// Sphere primitive
struct Sphere : public Primitive {
	float		radius;

	Sphere() {}
	Sphere(float r, const Transform &t, const Material &m)
			: Primitive(OBJECT_TYPE_SPHERE, t, m),
			radius(r) {}

	void write(Buffer &buffer) const override {
		buffer.push_back(aligned_vec4 {
			glm::vec4(transform.position, radius)
		});
	}
};


#endif
