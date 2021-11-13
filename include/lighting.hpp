#ifndef LIGHTING_H_
#define LIGHTING_H_

// Standard headers
#include <vector>

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "include/model.hpp"	// TODO: create a dedicated header/source for mesh
#include "include/shader.hpp"

namespace mercury {

namespace lighting {

// Material
struct Material {
	glm::vec3 color;
};

// Directional light
struct DirLight {
	glm::vec3 direction;

	glm::vec3 ambient;
	glm::vec3 diffuse;
	glm::vec3 specular;
};

// Point light
struct PointLight {
	glm::vec3 position;

	// TODO: separate into diffuse, spec, etc?
	glm::vec3 color;
};

// Lighting type: combine with binary or
// NOTE: Change the int-type for more types
enum Shading : uint8_t {
	COLOR_ONLY = 0,			// Should be mutually exclusive of all options
	DIRECTIONAL_PHONG = 1,
	POINT_PHONG = 2,
	FULL_PHONG = 3
};

// Daemon for lighting in 3D
class Daemon {
	// Compilation status for shaders
	bool _cmp_color_only = false;
	bool _cmp_phong = false;
	
	// List of lights
	struct {
		std::vector <DirLight> directional;
		std::vector <PointLight> point;
	} lights;

	// Compilers
	void _compile_color_only();
	void _compile_phong();

	// Renderers
	void _color_only(size_t);
	void _point_phong_only(size_t);
	void _dir_phong_only(size_t);
	void _full_phong(size_t);
public:
	// TODO: make all members private

	// All the types of shaders
	// NOTE: Compiled on a need-basis
	Shader color_only;
	Shader phong;

	// TODO: also need different shaders for different vertex formats (w or w/ normal, etc)

	// TODO: concatenate the vector structures struct {mesh, shading, mat...}

	// TODO: later generalize to allow
	// vertex buffer classes
	std::vector <Mesh *> objects;

	// Shading type for each object
	std::vector <Shading> shtypes;

	std::vector <Material> materials;

	// Shader uniforms
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	glm::vec3 view_position;

	// Adding lights
	void add_light(const DirLight &);
	void add_light(const PointLight &);

	// Add an object and specify the shading type
	void add_object(Mesh *, Material, Shading = FULL_PHONG);

	// Renders a frame
	void render();
};

}

}

#endif
