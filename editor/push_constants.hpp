#pragma once

// GLM headers
#include <glm/glm.hpp>

// Push constants for editor renderer
struct GBuffer_PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	int material_index;
        int texture_status;
};

struct Albedo_PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;

        glm::vec4 albedo;
        int has_albedo;
};

struct BoundingBox_PushConstants {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 projection;
        glm::vec4 color;
};

struct Highlight_PushConstants {
        glm::vec4 color;
        int material_index;
};
