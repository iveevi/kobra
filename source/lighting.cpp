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

	// The number of point lights is not being
	// set because depending on the shading mode,
	// they will have to be set to zero anyways
	_shaders.phong.use();
	_shaders.phong.set_vec3(direction, _lights.directional[i].direction);
	_shaders.phong.set_vec3(ambient, _lights.directional[i].ambient);
	_shaders.phong.set_vec3(diffuse, _lights.directional[i].diffuse);
	_shaders.phong.set_vec3(specular, _lights.directional[i].specular);
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

	_shaders.phong.use();
	_shaders.phong.set_vec3(color, _lights.point[i].color);
	_shaders.phong.set_vec3(position, _lights.point[i].position);
}

// Adding objects
void Daemon::add_object(Mesh *mesh, Material mat, Shading type)
{
	_robjs.push_back({mesh, mat, type});
	
	/* objects.push_back(mesh);
	materials.push_back(mat);
	shtypes.push_back(type); */

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
void Daemon::_color_only(const RenderObject &robj)
{
	// Check that the shader is compiled
	if (!_compiled.basic)
		Logger::fatal_error("COLOR_ONLY shader has not been compiled.");

	// Set color and render
	_shaders.basic.use();
	_shaders.basic.set_vec3("color", robj.material.color);
	robj.mesh->draw(_shaders.basic);
}

void Daemon::_phong_helper(const RenderObject &robj)
{
	// Check that the shader is compiled
	if (!_compiled.phong)
		Logger::fatal_error("POINT_PHONG shader has not been compiled.");
	
	// Set object material uniform
	_shaders.phong.use();
	_shaders.phong.set_vec3("color", robj.material.color);
}

void Daemon::_point_phong_only(const RenderObject &robj)
{
	_phong_helper(robj);
	_shaders.phong.set_int("ndir_lights", 0);	// Ignore directional lights
	_shaders.phong.set_int("npoint_lights", _lights.point.size());
	robj.mesh->draw(_shaders.phong);
}

void Daemon::_dir_phong_only(const RenderObject &robj)
{
	_phong_helper(robj);
	_shaders.phong.set_int("ndir_lights", _lights.directional.size());
	_shaders.phong.set_int("npoint_lights", 0);	// Ignore point lights
	robj.mesh->draw(_shaders.phong);
}

void Daemon::_full_phong(const RenderObject &robj)
{
	_phong_helper(robj);
	_shaders.phong.set_int("ndir_lights", _lights.directional.size());
	_shaders.phong.set_int("npoint_lights", _lights.point.size());
	robj.mesh->draw(_shaders.phong);
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

// Render a single object
void Daemon::_render_object(const RenderObject &robj)
{
	switch (robj.shtype) {
	case COLOR_ONLY:
		_color_only(robj);
		break;
	case POINT_PHONG:
		_point_phong_only(robj);
		break;
	case DIRECTIONAL_PHONG:
		_dir_phong_only(robj);
		break;
	case FULL_PHONG:
		_full_phong(robj);
		break;
	default:
		Logger::error() << "Undefined shading type ["
			<< (int) robj.shtype << "].\n";
		break;
	}
}

// Rendering a frame
void Daemon::render()
{
	_set_shader_uniforms();

	// Render each object
	for (const RenderObject &robj : _robjs)
		_render_object(robj);
	// for (size_t i = 0; i < objects.size(); i++)
	//	_render_object(i);
}

}

}
