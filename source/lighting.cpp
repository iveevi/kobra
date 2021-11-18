#include "include/logger.hpp"
#include "include/common.hpp"
#include "include/lighting.hpp"

namespace mercury {

namespace lighting {

// TODO: put these sources into files

// Source for color only	// TODO: this vert shader needs to be a basic 3d shader file...
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

// Constructors
Daemon::Daemon() {}

Daemon::Daemon(rendering::Daemon *rdam) : _rdaemon(rdam) {}

// Compiling each shader
void Daemon::_compile_color_only()
{
	if (_compiled.basic)
		return;

	// Compile the shader if not compiled
	_shaders.basic = Shader::from_source(
		color_only_vert,
		color_only_frag
	);
	_shaders.basic.set_name("basic_shader");

	// Set compilation flag
	_compiled.basic = true;
}

void Daemon::_compile_phong()
{
	if (_compiled.phong)
		return;
	
	// Compile the shader if not compiled
	_shaders.phong = Shader::from_source(
		phong_vert,
		phong_frag
	);
	_shaders.phong.set_name("phong_shader");

	// Set compilation flag
	_compiled.phong = true;
}

// Adding lights
void Daemon::add_light(const DirLight &light)
{
	// Get index for uniform
	size_t i = _lights.directional.size();
	_lights.directional.push_back(light);
	_compile_phong();
	
	// Set lighting uniforms
	std::string index = std::to_string(i);
	std::string direction = "dir_lights[" + index + "].direction";
	std::string ambient = "dir_lights[" + index + "].ambient";
	std::string diffuse = "dir_lights[" + index + "].diffuse";
	std::string specular = "dir_lights[" + index + "].specular";

	// Set the uniforms
	_shaders.phong.use();
	_shaders.phong.set_vec3(direction, _lights.directional[i].direction);
	_shaders.phong.set_vec3(ambient, _lights.directional[i].ambient);
	_shaders.phong.set_vec3(diffuse, _lights.directional[i].diffuse);
	_shaders.phong.set_vec3(specular, _lights.directional[i].specular);
	_shaders.phong.set_int("ndir_lights", _lights.directional.size());
}

void Daemon::add_light(const PointLight &light)
{
	// Get index for uniform
	size_t i = _lights.point.size();
	_lights.point.push_back(light);
	_compile_phong();

	// Set lighting uniforms
	std::string index = std::to_string(i);
	std::string color = "point_lights[" + index + "].color";
	std::string position = "point_lights[" + index + "].position";

	// Set the uniforms
	_shaders.phong.use();
	_shaders.phong.set_vec3(color, _lights.point[i].color);
	_shaders.phong.set_vec3(position, _lights.point[i].position);
	_shaders.phong.set_int("npoint_lights", _lights.point.size());
}

// Adding objects
void Daemon::add_object(Mesh *mesh, Shading type)
{
	_robjs.push_back({mesh, type});
	
	// Compiled the appropriate shader
	switch (type) {
	case COLOR_ONLY:
		_compile_color_only();
		_rdaemon->add(mesh, &_shaders.basic);
		break;
	case FULL_PHONG:
		_compile_phong();
		_rdaemon->add(mesh, &_shaders.phong);
		break;
	}
}

void Daemon::add_object(Mesh *mesh, glm::mat4 *model, Shading type)
{
	_robjs.push_back({mesh, type});
	
	// Compiled the appropriate shader
	switch (type) {
	case COLOR_ONLY:
		_compile_color_only();
		_rdaemon->add(mesh, &_shaders.basic, model);
		break;
	case FULL_PHONG:
		_compile_phong();
		_rdaemon->add(mesh, &_shaders.phong, model);
		break;
	}
}

// Set the common shader uniforms
void Daemon::_set_shader_uniforms()
{
	// Set scene relative uniforms for all shaders
	if (_compiled.basic) {
		_shaders.basic.use();
		_shaders.basic.set_mat4("model", uniforms.model);
		_shaders.basic.set_mat4("view", uniforms.view);
		_shaders.basic.set_mat4("projection", uniforms.projection);
	}

	if (_compiled.phong) {
		_shaders.phong.use();
		_shaders.phong.set_mat4("model", uniforms.model);
		_shaders.phong.set_mat4("view", uniforms.view);
		_shaders.phong.set_mat4("projection", uniforms.projection);
		_shaders.phong.set_vec3("view_position", uniforms.view_position);
	}
}

// Rendering a frame
void Daemon::light()
{
	_set_shader_uniforms();
}

}

}
