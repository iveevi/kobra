#ifndef COLLIDER_H_
#define COLLIDER_H_

// Standard headers
#include <utility>

// Engine headers
#include "include/transform.hpp"
#include "include/rendering.hpp"
#include "include/physics/aabb.hpp"

namespace mercury {

namespace physics {

// Collider abstract base class
struct Collider {
	// Aliases	
	using Interval = std::pair <float, float>;
	using Vertices = std::vector <glm::vec3>;

	// Members
	Transform *	transform = nullptr;

	// Constructors
	Collider(Transform *);

	// TODO: some way check if the collider has velocity

	// Make the AABB (using the virtual support function)
	AABB aabb() const;

	virtual glm::vec3 support(const glm::vec3 &) const = 0;
};

// Standard colliders
class BoxCollider : public Collider {
	Vertices	_vertices;
public:
	// Members
	glm::vec3 center;		// NOTE: These are local coordinates
	glm::vec3 size;			// No need for an up vector, comes from transform

	// Constructor
	BoxCollider(const glm::vec3 &, Transform *);
	BoxCollider(const glm::vec3 &, const glm::vec3 &, Transform *);

	// Methods
	void annotate(rendering::Daemon &, Shader *) const;

	// Virtual overrides
	virtual glm::vec3 support(const glm::vec3 &) const override;
};

// Srtucture detailing the result of a collision
struct Collision {
	glm::vec3	mtv;
	bool		colliding;
};

// Simplex structure for the GJK algorithm
// TODO: separate into source
class Simplex {
	glm::vec3	_points[4];
	size_t		_size;
public:
	Simplex() : _size(0) {}

	// Assign with initializer list
	Simplex &operator=(std::initializer_list <glm::vec3> il) {
		_size = 0;
		for (const glm::vec3 &pt : il)
			_points[_size++] = pt;
		
		return *this;
	}

	// Size
	size_t size() const {
		return _size;
	}

	// Push vector
	void push(const glm::vec3 &v) {
		// Cycle the points
		_points[3] = _points[2];
		_points[2] = _points[1];
		_points[1] = _points[0];
		_points[0] = v;

		// Cap the index
		_size = std::min(_size + 1, 4UL);
	}

	// Indexing
	const glm::vec3 &operator[](size_t index) const {
		return _points[index];
	}

	// Vertices
	Collider::Vertices vertices() const {
		Collider::Vertices v;
		for (size_t i = 0; i < _size; i++)
			v.push_back(_points[i]);
		return v;
	}
};

// Intersection between two colliders
// TODO: remove extra debugging parameters
Collision intersects(Collider *, Collider *);

}

}

#endif