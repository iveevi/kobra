#ifndef LINALG_H_
#define LINALG_H_

// GLM headers
#include <glm/glm.hpp>

namespace mercury {

namespace math {

// Project v over u
inline glm::vec3 project(const glm::vec3 &v, const glm::vec3 &u)
{
	return u * glm::dot(v, u)/glm::dot(u, u);
}

// Get the length of the projection
inline float projection_length(const glm::vec3 &pt, const glm::vec3 &naxis)
{
	return glm::length(math::project(pt, naxis));
}

// Spherical coordinates to cartesian
inline glm::vec3 sph_to_cart(float xy, float xz)
{
	float xr_rad = cos(xy);
	return glm::vec3(
		xr_rad * cos(xz),
		sin(xy),
		xr_rad * sin(xz)
	);
}

}

}

#endif