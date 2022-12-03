#include <vector>
#include <glm/glm.hpp>
#include "include/core/interpolation.hpp"

/*
const std::vector <glm::vec3> CAMERA_POSITIONS {
	{99.09, 19.95, 21.30},
	{0.77, 18.36, -36.41}
};

const std::vector <glm::vec3> CAMERA_ROTATIONS {
	{-0.01, 1.04, 0.00},
	{-0.01, 1.04, 0.00}
};
*/

/* const std::vector <glm::vec3> CAMERA_POSITIONS {
	{-4.69, 16.34, 18.22},
	{4.74, 16.34, 18.22},
	{4.74, 16.34, 30.96},
	{4.74, 5.54, 30.96},
	{-6.26, 5.54, 30.96},
};

const std::vector <glm::vec3> CAMERA_ROTATIONS {
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
}; */

const std::vector <glm::vec3> CAMERA_POSITIONS {
	{31.19, 36.55, -9.84},
	{122.12, 35.10, -9.09}
};

const std::vector <glm::vec3> CAMERA_ROTATIONS {
	{-0.02, -1.58, 0.00},
	{-0.02, -1.58, 0.00}
};

const std::vector <float> CAMERA_TIMES {
	0, 5
};

const kobra::core::Sequence <glm::vec3> CAMERA_POSITION_SEQUENCE {
	CAMERA_POSITIONS,
	CAMERA_TIMES
};

const kobra::core::Sequence <glm::vec3> CAMERA_ROTATION_SEQUENCE {
	CAMERA_ROTATIONS,
	CAMERA_TIMES
};
