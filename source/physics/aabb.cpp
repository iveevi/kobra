// Engine headers
#include "include/mesh/cuboid.hpp"
#include "include/physics/aabb.hpp"

namespace mercury {

namespace physics {

// Method
bool AABB::intersects(const AABB &other) const
{
	// Edges
	glm::vec3 emax = center + size/2.0f;
	glm::vec3 emin = center - size/2.0f;
	
	glm::vec3 omax = other.center + other.size/2.0f;
	glm::vec3 omin = other.center - other.size/2.0f;

	/* Logger::notify() << "AABB coordinates:\n";
	Logger::notify() << "\temax = " << emax.x << ", " << emax.y << ", " << emax.z << "\n";
	Logger::notify() << "\temin = " << emin.x << ", " << emin.y << ", " << emin.z << "\n";
	Logger::notify() << "\tomax = " << omax.x << ", " << omax.y << ", " << omax.z << "\n";
	Logger::notify() << "\tomin = " << omin.x << ", " << omin.y << ", " << omin.z << "\n";

	Logger::notify() << "\tcenter = " << center.x << ", " << center.y << ", " << center.z << "\n";
	Logger::notify() << "\tsize = " << size.x << ", " << size.y << ", " << size.z << "\n";

	Logger::notify() << "\tother.center = " << other.center.x << ", " << other.center.y << ", " << other.center.z << "\n";
	Logger::notify() << "\tother.size = " << other.size.x << ", " << other.size.y << ", " << other.size.z << "\n"; */

	// Check all axes
	return (omin.x <= emax.x && omax.x >= emin.x) &&
		(omin.y <= emax.y && omax.y >= emin.y) &&
		(omin.z <= emax.z && omax.z >= emin.z);
}

// Annotate
void AABB::annotate(rendering::Daemon &rdam, Shader *shader)
{
	static const glm::vec3 color = {1.0, 1.0, 0.5};

	// Transform only takes the translation and size
	SVA3 *box = new SVA3(mesh::wireframe_cuboid(center, size));
	box->color = color;

	rdam.add(box, shader);
}

// Function
bool intersects(const AABB &a, const AABB &b)
{
	return a.intersects(b);
}

}

}