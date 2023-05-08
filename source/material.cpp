// Standard headers
#include <thread>

// Engine headers
#include "../include/material.hpp"
#include "../include/profiler.hpp"

namespace kobra {

// Global material list
// std::vector <Material> Material::all;
// Material::Daemon Material::daemon;

// Properties
bool Material::has_albedo() const
{
	return !(albedo_texture.empty() || albedo_texture == "0");
}

bool Material::has_normal() const
{
	return !(normal_texture.empty() || normal_texture == "0");
}

bool Material::has_specular() const
{
	return !(specular_texture.empty() || specular_texture == "0");
}

bool Material::has_emission() const
{
	return !(emission_texture.empty() || emission_texture == "0");
}

bool Material::has_roughness() const
{
	return !(roughness_texture.empty() || roughness_texture == "0");
}

// Serialize to buffer
void Material::serialize(std::vector <aligned_vec4> &buffer) const
{
	int i_type = static_cast <int> (type);
	float f_type = *reinterpret_cast <float *> (&i_type);

	buffer.push_back(aligned_vec4(diffuse, f_type));
	buffer.push_back(aligned_vec4(
		{refraction, albedo_texture.empty(), normal_texture.empty(), 0}
	));
}

}
