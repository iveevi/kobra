#ifndef PRIMITIVE_H_
#define PRIMITIVE_H_

// Engine headers
#include "bbox.hpp"
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

	// Extract bounding boxes
	virtual void extract_bboxes(std::vector <mercury::BoundingBox> &bboxes) const = 0;

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
	Triangle(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c)
			: a(a), b(b), c(c) {}
	Triangle(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c,
			const Material &m)
			: Primitive(OBJECT_TYPE_TRIANGLE, Transform(), m),
			a(a), b(b), c(c) {}

	void write(Buffer &buffer) const override {
		buffer.push_back(aligned_vec4(a));
		buffer.push_back(aligned_vec4(b));
		buffer.push_back(aligned_vec4(c));
	}

	void extract_bboxes(std::vector <mercury::BoundingBox> &bboxes) const override {
		// Get min and max coordinates 
		// TODO: account for transform
		glm::vec3 min = glm::min(a, glm::min(b, c));
		glm::vec3 max = glm::max(a, glm::max(b, c));

		// Create bounding box
		glm::vec3 center = (min + max) / 2.0f;
		glm::vec3 size = (max - min) / 2.0f;

		// Push bounding box
		bboxes.push_back(mercury::BoundingBox(center, size));
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

	void extract_bboxes(std::vector <mercury::BoundingBox> &bboxes) const override {
		// Create bounding box
		glm::vec3 center = transform.position;
		glm::vec3 size = glm::vec3(radius);

		// Push bounding box
		bboxes.push_back(mercury::BoundingBox(center, size));
	}
};


#endif
