#include "include/material.hpp"

namespace kobra {

// Properties
bool Material::has_albedo() const
{
	return !(diffuse_texture.empty() || diffuse_texture == "0");
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

}
