// Engine headers
#include "include/common.hpp"
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

// TODO: use cres.basic
void BoxCollider::annotate(rendering::Daemon &rdam, Shader *shader) const
{
	static const glm::vec3 color = {1.0, 0.5, 1.0};

	glm::mat4 model = transform->model();
	glm::vec3 up = {0, 1, 0};
	glm::vec3 right = {1, 0, 0};

	SVA3 *box = new SVA3(mesh::wireframe_cuboid(center, size, up, right));
	box->color = color;

	rdam.add(box, shader, transform);
}

AABB BoxCollider::aabb() const
{
	glm::mat4 model = transform->model();
	glm::vec3 ncenter = glm::vec3(model * glm::vec4(center, 1.0f));
	glm::vec3 nsize = size * transform->scale/2.0f;			// Only consdider scale

	// All vertices
	glm::vec3 nright = glm::normalize(glm::vec3(model * glm::vec4 {1.0, 0.0, 0.0, 0.0}));
	glm::vec3 nup = glm::normalize(glm::vec3(model * glm::vec4 {0.0, 1.0, 0.0, 0.0}));
	glm::vec3 nforward = glm::normalize(glm::vec3(model * glm::vec4 {0.0, 0.0, 1.0, 0.0}));

	std::vector <glm::vec3> vertices {
		ncenter + nright * nsize.x + nup * nsize.y + nforward * nsize.z,
		ncenter + nright * nsize.x + nup * nsize.y - nforward * nsize.z,
		ncenter + nright * nsize.x - nup * nsize.y + nforward * nsize.z,
		ncenter + nright * nsize.x - nup * nsize.y - nforward * nsize.z,
		ncenter - nright * nsize.x + nup * nsize.y + nforward * nsize.z,
		ncenter - nright * nsize.x + nup * nsize.y - nforward * nsize.z,
		ncenter - nright * nsize.x - nup * nsize.y + nforward * nsize.z,
		ncenter - nright * nsize.x - nup * nsize.y - nforward * nsize.z
	};

	// Max and min axial coordinates
	float max_x, max_y, max_z;
	float min_x, min_y, min_z;

	max_x = max_y = max_z = -std::numeric_limits <float> ::max();
	min_x = min_y = min_z = std::numeric_limits <float> ::max();

	// Find max and min axial coordinates
	/* Logger::notify() << "Calculating min/max:\n";
	Logger::notify() << "--> mins: <" << min_x << ", " << min_y << ", " << min_z << ">\n";
	Logger::notify() << "--> maxs: <" << max_x << ", " << max_y << ", " << max_z << ">\n"; */
	
	for (auto &v : vertices) {
		// Logger::notify() << "\tVertex: " << v.x << ", " << v.y << ", " << v.z << "\n";
		if (v.x > max_x) max_x = v.x;
		if (v.y > max_y) max_y = v.y;
		if (v.z > max_z) max_z = v.z;

		if (v.x < min_x) min_x = v.x;
		if (v.y < min_y) min_y = v.y;
		if (v.z < min_z) min_z = v.z;
	}

	// Calculate AABB positions
	glm::vec3 box_center {
		(min_x + max_x) / 2.0f,
		(min_y + max_y) / 2.0f,
		(min_z + max_z) / 2.0f
	};

	glm::vec3 box_size {
		max_x - min_x,
		max_y - min_y,
		max_z - min_z
	};

	/* Logger::notify() << "--> mins: <" << min_x << ", " << min_y << ", " << min_z << ">\n";
	Logger::notify() << "--> maxs: <" << max_x << ", " << max_y << ", " << max_z << ">\n";

	Logger::notify() << "box_center = " << box_center.x << ", " << box_center.y << ", " << box_center.z << "\n";
	Logger::notify() << "box_size = " << box_size.x << ", " << box_size.y << ", " << box_size.z << "\n";

	Logger::notify() << "real center = " << center.x << ", " << center.y << ", " << center.z << "\n";
	Logger::notify() << "real size = " << size.x << ", " << size.y << ", " << size.z << "\n"; */

	// Return AABB
	return AABB {box_center, box_size};
}

}

}