// Engine headers
#include "include/physics/collider.hpp"
#include "include/mesh/cuboid.hpp"

namespace mercury {

namespace physics {

// Constructors
Collider::Collider(Transform *tptr) : transform(tptr) {}

// Intersection between two colliders
bool intersects(Collider *a, Collider *b)
{
	// TODO: should aabb be cached?
	return intersects(a->aabb(), b->aabb());
}

// Box Collider
BoxCollider::BoxCollider(const glm::vec3 &s, Transform *tptr)
		: Collider(tptr), center({0, 0, 0}), size(s) {}

BoxCollider::BoxCollider(const glm::vec3 &c, const glm::vec3 &s, Transform *tptr)
		: Collider(tptr), center(c), size(s) {}

// TODO: use cres.ann_shader
void BoxCollider::annotate(lighting::Daemon &ldam) const
{
	glm::mat4 model = transform->model();
	glm::vec3 up = glm::vec3(model * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
	glm::vec3 right = glm::vec3(model * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));

	SVA3 *box = new SVA3(mesh::wireframe_cuboid(center, size, up, right));
	// ldam.add_object(box, transform, lighting::COLOR_ONLY);
}

AABB BoxCollider::aabb() const
{
	glm::mat4 model = transform->model();
	glm::vec3 center = glm::vec3(model * glm::vec4(center, 1.0f));
	glm::vec3 size = glm::vec3(model * glm::vec4(size, 1.0f));

	// All vertices
	std::vector <glm::vec3> vertices {
		center + size,
		center + glm::vec3(size.x, -size.y, size.z),
		center + glm::vec3(-size.x, -size.y, size.z),
		center - size,
		center + glm::vec3(size.x, -size.y, -size.z),
		center + glm::vec3(size.x, size.y, size.z),
		center + glm::vec3(-size.x, size.y, size.z),
		center + glm::vec3(-size.x, size.y, -size.z),
		center + glm::vec3(size.x, size.y, -size.z)
	};

	// Max and min axial coordinates
	float max_x, max_y, max_z = 0.0f;
	float min_x, min_y, min_z = std::numeric_limits <float>::max();

	// Find max and min axial coordinates
	for (auto &v : vertices) {
		if (v.x > max_x) max_x = v.x;
		if (v.y > max_y) max_y = v.y;
		if (v.z > max_z) max_z = v.z;

		if (v.x < min_x) min_x = v.x;
		if (v.y < min_y) min_y = v.y;
		if (v.z < min_z) min_z = v.z;
	}

	// Calculate AABB positions
	glm::vec3 ncenter {
		(min_x + max_x) / 2.0f,
		(min_y + max_y) / 2.0f,
		(min_z + max_z) / 2.0f
	};

	glm::vec3 nsize {
		max_x - min_x,
		max_y - min_y,
		max_z - min_z
	};
	

	// Return AABB
	return AABB {ncenter, nsize};
}

}

}