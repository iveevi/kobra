#include "include/logger.hpp"
#include "include/common.hpp"
#include "include/lighting.hpp"

namespace mercury {

namespace lighting {

// Source for color only
const char *color_only_vert = R"(
#version 330 core

layout (location = 0) in vec3 v_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * vec4(v_pos, 1.0);
}
)";

const char *color_only_frag = R"(
// Outputs
out vec4 frag_color;

// Uniforms
uniform vec3 color;

void main()
{
	frag_color = vec4(color, 1.0);
}
)";

const char *phong_vert = R"(
#version 330 core

layout (location = 0) in vec3 v_pos;
layout (location = 1) in vec3 v_normal;

out vec3 normal;
out vec3 frag_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	vec4 pos = vec4(v_pos, 1.0);
	gl_Position = projection * view * model * pos;
	frag_pos = vec3(model * pos);

	// TODO: this should be done on the cpu because its intensive
	normal = mat3(transpose(inverse(model))) * v_normal;
}
)";

// TODO: create different version based on the number of lights
//	using some preproc system
const char *phong_frag = R"(
#include point_light
#include dir_light

in vec3 normal;
in vec3 frag_pos;

out vec4 frag_color;

// TODO: put these into a material struct
uniform vec3 color;

// View position
uniform vec3 view_position;

#define NLIGHTS 4

// Light arrays
uniform PointLight point_lights[NLIGHTS];
uniform DirLight dir_lights[NLIGHTS];

// Number of lights (0 if not using the light mode)
uniform int npoint_lights;
uniform int ndir_lights;

void main()
{
	// Equivalent to sum variable
	vec3 result = vec3(0.0, 0.0, 0.0);

	// Point lights
	for (int i = 0; i < npoint_lights; i++)
		result += point_light_contr(point_lights[i], color, frag_pos, view_position, normal);
	
	// Dir lights
	for (int i = 0; i < ndir_lights; i++)
		result += dir_light_contr(dir_lights[i], color, frag_pos, view_position, normal);

	// Return the color
	frag_color = vec4(result, 1.0);
}
)";

// Compiling each shader
void Daemon::_compile_color_only()
{
	if (_cmp_color_only)
		return;

	// Compile the shader if not compiled
	color_only = Shader::from_source(
		color_only_vert,
		color_only_frag
	);
	color_only.set_name("color_only_shader");

	// Set compilation flag
	_cmp_color_only = true;
}

void Daemon::_compile_phong()
{
	if (_cmp_phong)
		return;
	
	// Compile the shader if not compiled
	phong = Shader::from_source(
		phong_vert,
		phong_frag
	);
	phong.set_name("phong_shader");

	// Set compilation flag
	_cmp_phong = true;
}

// Adding lights
void Daemon::add_light(const DirLight &light)
{
	// Get index for uniform
	size_t i = lights.directional.size();
	lights.directional.push_back(light);
	_compile_phong();
	
	// Set lighting uniforms
	std::string index = std::to_string(i);
	std::string direction = "dir_lights[" + index + "].direction";
	std::string ambient = "dir_lights[" + index + "].ambient";
	std::string diffuse = "dir_lights[" + index + "].diffuse";
	std::string specular = "dir_lights[" + index + "].specular";

	// The number of point lights is not being
	// set because depending on the shading mode,
	// they will have to be set to zero anyways
	phong.use();
	phong.set_vec3(direction, lights.directional[i].direction);
	phong.set_vec3(ambient, lights.directional[i].ambient);
	phong.set_vec3(diffuse, lights.directional[i].diffuse);
	phong.set_vec3(specular, lights.directional[i].specular);
}

void Daemon::add_light(const PointLight &light)
{
	// Get index for uniform
	size_t i = lights.point.size();
	lights.point.push_back(light);
	_compile_phong();

	// Set lighting uniforms
	std::string index = std::to_string(i);
	std::string color = "point_lights[" + index + "].color";
	std::string position = "point_lights[" + index + "].position";

	phong.use();
	phong.set_vec3(color, lights.point[i].color);
	phong.set_vec3(position, lights.point[i].position);
}

// Adding objects
void Daemon::add_object(Mesh *mesh, Material mat, Shading type)
{
	objects.push_back(mesh);
	materials.push_back(mat);
	shtypes.push_back(type);

	// Compiled the appropriate shader
	switch (type) {
	case COLOR_ONLY:
		_compile_color_only();
		break;
	case POINT_PHONG:
	case DIRECTIONAL_PHONG:
	case FULL_PHONG:
		_compile_phong();
		break;
	}
}

// Render object with color only
void Daemon::_color_only(size_t obj)
{
	// Check that the shader is compiled
	if (!_cmp_color_only)
		Logger::fatal_error("COLOR_ONLY shader has not been compiled.");

	// Set color and render
	color_only.use();
	color_only.set_vec3("color", materials[obj].color);
	objects[obj]->draw(color_only);
}

// TODO: add a helper function that does all the same stuff except sizes
void Daemon::_point_phong_only(size_t obj)
{
	// Check that the shader is compiled
	if (!_cmp_phong)
		Logger::fatal_error("POINT_PHONG shader has not been compiled.");

	// Render the object
	phong.use();
	phong.set_int("ndir_lights", 0);	// Ignore directional lights
	phong.set_int("npoint_lights", lights.point.size());
	phong.set_vec3("color", materials[obj].color);
	objects[obj]->draw(phong);
}

void Daemon::_dir_phong_only(size_t obj)
{
	// Check that the shader is compiled
	if (!_cmp_phong)
		Logger::fatal_error("POINT_PHONG shader has not been compiled.");

	// Render the object
	phong.use();
	phong.set_int("ndir_lights", lights.directional.size());
	phong.set_int("npoint_lights", 0);	// Ignore point lights
	phong.set_vec3("color", materials[obj].color);
	objects[obj]->draw(phong);
}

void Daemon::_full_phong(size_t obj)
{
	// Check that the shader is compiled
	if (!_cmp_phong)
		Logger::fatal_error("POINT_PHONG shader has not been compiled.");

	// Render the object
	phong.use();
	phong.set_int("ndir_lights", lights.directional.size());
	phong.set_int("npoint_lights", lights.point.size());
	phong.set_vec3("color", materials[obj].color);
	objects[obj]->draw(phong);
}

// Rendering a frame
void Daemon::render()
{
	// Set scene relative uniforms for all shaders
	// TODO: do in private helper function
	if (_cmp_color_only) {
		color_only.use();
		color_only.set_mat4("model", model);
		color_only.set_mat4("view", view);
		color_only.set_mat4("projection", projection);
	}

	if (_cmp_phong) {
		phong.use();
		phong.set_mat4("model", model);
		phong.set_mat4("view", view);
		phong.set_mat4("projection", projection);
		phong.set_vec3("view_position", view_position);
	}

	// Render each object
	for (size_t i = 0; i < objects.size(); i++) {
		switch (shtypes[i]) {
		case COLOR_ONLY:
			_color_only(i);
			break;
		case POINT_PHONG:
			_point_phong_only(i);
			break;
		case DIRECTIONAL_PHONG:
			_dir_phong_only(i);
			break;
		case FULL_PHONG:
			_full_phong(i);
			break;
		default:
			Logger::error() << "Undefined shading type ["
				<< (int) shtypes[i] << "].\n";
			break;
		}
	}
}

}

}
