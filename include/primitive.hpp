#ifndef PRIMITIVE_H_
#define PRIMITIVE_H_

// Standard headers
#include <fstream>

// Engine headers
#include "bbox.hpp"
#include "core.hpp"
#include "material.hpp"
#include "transform.hpp"
#include "types.hpp"
#include "world_update.hpp"

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

	// Count number of primitives
	virtual uint count() const = 0;

	// Save to a file
	virtual void save(std::ofstream &file) const = 0;

	// Wrappers around virtual functions
	void save_to_file(std::ofstream &file) const {
		// TODO: save as binary at some point?
		save(file);

		// Save transform and material
		// file << "\tTransform: " << transform.position << "\n";
	}

	// Write data to aligned_vec4 buffer (inherited)
	virtual void write(mercury::WorldUpdate &) const = 0;

	// Extract bounding boxes
	virtual void extract_bboxes(std::vector <mercury::BoundingBox> &bboxes) const = 0;

	// Write header
	void write_header(mercury::WorldUpdate &wu, uint mati) const {
		float index = *reinterpret_cast <float *> (&mati);

		// Push ID and material, then everything else
		wu.bf_objs->push_back(aligned_vec4 {
			glm::vec4(id, index, 0.0, 0.0)
		});
	}

	// Write full object data
	// TODO: const?
	virtual void write_object(mercury::WorldUpdate &wu) {
		// Deal with material
		uint mati = wu.bf_mats->push_size();
		material.write_material(wu);

		/* float index = *reinterpret_cast <float *> (&mati);

		// Push ID and material, then everything else
		wu.bf_objs->push_back(aligned_vec4 {
			glm::vec4(id, index, 0.0, 0.0)
		}); */
		write_header(wu, mati);
		this->write(wu);
	}

	// Write full object data, but takes index to material
	virtual void write_object_mati(mercury::WorldUpdate &wu, uint mati) {
		/* Deal with material
		float index = *reinterpret_cast <float *> (&mati);

		// Push ID and material, then everything else
		wu.bf_objs->push_back(aligned_vec4 {
			glm::vec4(id, index, 0.0, 0.0)
		}); */

		write_header(wu, mati);
		this->write(wu);
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

	uint count() const override { return 1; }

	// Save to file
	void save(std::ofstream &file) const override {
		// Header for object
		file << "Triangle\n";

		// Write positions in binary
		file << "\tpositions:";
		file.write(reinterpret_cast <const char *> (&a), sizeof(glm::vec3));
		file.write(reinterpret_cast <const char *> (&b), sizeof(glm::vec3));
		file.write(reinterpret_cast <const char *> (&c), sizeof(glm::vec3));
		file << "\n";
	}

	void write(mercury::WorldUpdate &wu) const override {
		/* wu.bf_objs->push_back(aligned_vec4(a));
		wu.bf_objs->push_back(aligned_vec4(b));
		wu.bf_objs->push_back(aligned_vec4(c)); */
		
		uint index = wu.bf_verts->push_size();
		wu.bf_verts->push_back(aligned_vec4(a));
		wu.bf_verts->push_back(aligned_vec4(b));
		wu.bf_verts->push_back(aligned_vec4(c));

		uint ia = index, ib = index + 1, ic = index + 2;
		float *iaf = reinterpret_cast <float *> (&ia);
		float *ibf = reinterpret_cast <float *> (&ib);
		float *icf = reinterpret_cast <float *> (&ic);

		wu.bf_objs->push_back(aligned_vec4 {
			glm::vec4(*iaf, *ibf, *icf, 0.0)
		});
	}

	// Write, but indices are given
	void write_indexed(mercury::WorldUpdate &wu,
			uint a, uint b, uint c, uint mati) const {
		// Header from parent
		write_header(wu, mati);
		
		float *iaf = reinterpret_cast <float *> (&a);
		float *ibf = reinterpret_cast <float *> (&b);
		float *icf = reinterpret_cast <float *> (&c);

		wu.bf_objs->push_back(aligned_vec4 {
			glm::vec4(*iaf, *ibf, *icf, 0.0)
		});		
	}

	void extract_bboxes(std::vector <mercury::BoundingBox> &bboxes) const override {
		// Get min and max coordinates 
		// TODO: account for transform
		glm::vec3 min = glm::min(a, glm::min(b, c));
		glm::vec3 max = glm::max(a, glm::max(b, c));

		// Push bounding box
		bboxes.push_back(mercury::BoundingBox(min, max));
	}
};

// Sphere primitive
struct Sphere : public Primitive {
	float		radius;

	Sphere() {}
	Sphere(float r, const Transform &t, const Material &m)
			: Primitive(OBJECT_TYPE_SPHERE, t, m),
			radius(r) {}

	uint count() const override { return 1; }

	// Save to file
	void save(std::ofstream &file) const override {
		// Header for object
		file << "Sphere\n";

		// Write radius in binary
		file << "\tradius: ";
		file.write(reinterpret_cast <const char *> (&radius), sizeof(float));
		file << "\n";
	}

	void write(mercury::WorldUpdate &wu) const override {
		wu.bf_objs->push_back(aligned_vec4 {
			glm::vec4(transform.position, radius)
		});
	}

	void extract_bboxes(std::vector <mercury::BoundingBox> &bboxes) const override {
		// Create bounding box
		glm::vec3 min = transform.position - glm::vec3(radius);
		glm::vec3 max = transform.position + glm::vec3(radius);

		// Push bounding box
		bboxes.push_back(mercury::BoundingBox(min, max));
	}
};


#endif
