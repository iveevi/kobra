#include "include/common.hpp"

// Overload printing glm::vec3
std::ostream &operator<<(std::ostream &os, const glm::vec3 &vec)
{
	return os << "<" << vec.x << ", " << vec.y
		<< ", " << vec.z << ">";
}
