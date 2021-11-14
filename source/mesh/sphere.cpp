// Standard headers
#include <vector>

// Engine headers
#include "include/mesh/sphere.hpp"

namespace mercury {

namespace mesh {

// Generate wireframe sphere
// TODO: keep somewhere else
static void _add_vec3(std::vector <float> &vertices, const glm::vec3 &vec)
{
        vertices.push_back(vec.x);
        vertices.push_back(vec.y);
        vertices.push_back(vec.z);
}

SVA3 wireframe_sphere(const glm::vec3 &center, float radius, int divisions)
{
        // Constants
        const int divs = (divisions + 3) & (-3);        // Round to next multiple of 4
	const float slice = 2 * acos(-1)/divs;

	// TODO: add helper method for axes and use lambda with theta

        // Array of vertices
        std::vector <float> vertices;   // TODO: preallocate the space

	// xy axial ring
	for (int i = 0; i <= divs; i++) {
		float theta = slice * i;
		glm::vec3 off {cos(theta), sin(theta), 0.0};
		off *= radius;
		_add_vec3(vertices, off + center);
	}
	
	// xz axial ring
	for (int i = 0; i <= 1.25 * divs; i++) {	// 1.25 to go the intersection with the next ring
		float theta = slice * i;
		glm::vec3 off {cos(theta), 0.0, sin(theta)};
		off *= radius;
		_add_vec3(vertices, off + center);
	}
	
	// yz axial ring
	for (int i = 0; i <= divs; i++) {
		float theta = slice * i + acos(-1)/2;
		glm::vec3 off {0.0, cos(theta), sin(theta)};
		off *= radius;
		_add_vec3(vertices, off + center);
	}
        
        // Return vertex array object
        return SVA3(vertices);
}

}

}