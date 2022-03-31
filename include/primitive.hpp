#ifndef PRIMITIVE_H_
#define PRIMITIVE_H_

// Standard headers
#include <fstream>

// Engine headers
#include "bbox.hpp"
#include "core.hpp"
#include "material.hpp"
#include "object.hpp"
#include "renderable.hpp"
#include "transform.hpp"
#include "types.hpp"
#include "world_update.hpp"

namespace kobra {

namespace raytracing {

// Primitive structures
struct Primitive : virtual public Object, virtual public Renderable {
public:
	static constexpr char object_type[] = "RT Primitive";
protected:
	float		_id = OBJECT_TYPE_NONE;
public:
	// Default constructor
	Primitive() = default;

	// Constructor
	Primitive(float x, const Transform &t, const Material &m)
			: Object(object_type, t),
			Renderable(m),
			_id(x) {}

	// Virtual object destructor
	virtual ~Primitive() {}

	// Count number of primitives
	virtual uint count() const = 0;

	// Save to a file
	virtual void save(std::ofstream &file) const = 0;

	// Set material
	void set_material(const Material &m) {
		_material = m;
	}

	// Wrappers around virtual functions
	void save_to_file(std::ofstream &file) const {
		// TODO: save as binary at some point?
		save(file);

		// Save transform and material
		// file << "\tTransform: " << transform.position << "\n";
	}

	// Write data to aligned_vec4 buffer (inherited)
	virtual void write(kobra::WorldUpdate &) const = 0;

	// Extract bounding boxes
	virtual void extract_bboxes(std::vector <kobra::BoundingBox> &,
			const glm::mat4 & = glm::mat4 {1.0}) const = 0;

	// Write header
	void write_header(kobra::WorldUpdate &wu, uint mati, uint tati) const {
		float material_index = *reinterpret_cast <float *> (&mati);
		float transform_index = *reinterpret_cast <float *> (&tati);

		// Push ID and material, then everything else
		wu.bf_objs->push_back(aligned_vec4 {
			glm::vec4(_id, material_index, transform_index, 0.0)
		});
	}

	// Write full object data
	// TODO: const?
	virtual void write_object(kobra::WorldUpdate &wu) {
		// Deal with material
		uint mati = wu.bf_mats->push_size();
		_material.write_material(wu);

		// Deal with transform
		uint tati = wu.bf_trans->push_size();
		wu.bf_trans->push_back(_transform.matrix());

		write_header(wu, mati, tati);
		this->write(wu);
	}

	// Write full object data, but takes indices
	// TODO: is this needed anymore?
	virtual void write_object_mati(kobra::WorldUpdate &wu, uint mati, uint tati) {
		write_header(wu, mati, tati);
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

	void write(kobra::WorldUpdate &wu) const override {
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
	void write_indexed(kobra::WorldUpdate &wu,
			uint a, uint b, uint c, uint mati, uint tati) const {
		// Header from parent
		write_header(wu, mati, tati);

		float *iaf = reinterpret_cast <float *> (&a);
		float *ibf = reinterpret_cast <float *> (&b);
		float *icf = reinterpret_cast <float *> (&c);

		wu.bf_objs->push_back(aligned_vec4 {
			glm::vec4(*iaf, *ibf, *icf, 0.0)
		});
	}

	void extract_bboxes(std::vector <kobra::BoundingBox> &bboxes, const glm::mat4 &parent) const override {
		// Get min and max coordinates
		// TODO: account for transform
		glm::mat4 m = parent * _transform.matrix();
		glm::vec3 ta = m * glm::vec4(a, 1.0);
		glm::vec3 tb = m * glm::vec4(b, 1.0);
		glm::vec3 tc = m * glm::vec4(c, 1.0);

		glm::vec3 min = glm::min(ta, glm::min(tb, tc));
		glm::vec3 max = glm::max(ta, glm::max(tb, tc));

		// Push bounding box
		bboxes.push_back(BoundingBox {min, max});
	}
};

// Sphere primitive
struct Sphere : public Primitive {
	static constexpr char object_type[] = "RT Sphere";

	// Members
	float		radius;

	Sphere() {}
	Sphere(float r, const Transform &t, const Material &m)
			: Object(object_type, t),
			Primitive(OBJECT_TYPE_SPHERE, t, m),
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

	void write(kobra::WorldUpdate &wu) const override {
		wu.bf_objs->push_back(aligned_vec4 {
			glm::vec4(radius)
		});
	}

	void extract_bboxes(std::vector <kobra::BoundingBox> &bboxes, const glm::mat4 &parent) const override {
		// Create bounding box
		glm::mat4 m = parent * _transform.matrix();
		glm::vec3 pos = m * glm::vec4(0.0, 0.0, 0.0, 1.0);
		glm::vec3 min = pos - glm::vec3(radius);
		glm::vec3 max = pos + glm::vec3(radius);

		// Push bounding box
		bboxes.push_back(BoundingBox {min, max});
	}
};

}

}

#endif
