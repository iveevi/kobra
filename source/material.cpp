// Standard headers
#include <thread>

// Engine headers
#include "../include/material.hpp"
#include "../include/profiler.hpp"

namespace kobra {

// Properties
bool Material::has_albedo() const
{
	return !(albedo_texture.empty() || albedo_texture == "0");
}

bool Material::has_normal() const
{
	return !(normal_texture.empty() || normal_texture == "0");
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

// Save material to file
void Material::save(std::ofstream &file) const
{
	file << "[MATERIAL]\n";
	file << "diffuse=" << diffuse.x << " " << diffuse.y << " " << diffuse.z << std::endl;
	file << "Ks=" << specular.x << " " << specular.y << " " << specular.z << std::endl;
	file << "shading_type=" << shading_str(type) << std::endl;
	file << "index_of_refraction=" << refraction << std::endl;
	// file << "extinction_coefficient=" << refr_k << std::endl;

	file << "albedo_texture=" << (albedo_texture.empty() ? "0" : albedo_texture) << std::endl;
	file << "normal_texture=" << (normal_texture.empty() ? "0" : normal_texture) << std::endl;
}

// Read material from file
Material Material::from_file(std::ifstream &file, const std::string &scene_file, bool &success)
{
	// Material to return
	Material mat;

	// Default success to false
	success = false;

	// TODO: field reader and writer class...
	// 	template parameter, with a lambda to
	// 	process and reutrn the value
	std::string line;

	// Read Kd
	glm::vec3 Kd;
	std::getline(file, line);

	// Ensure correct header ("Kd=")
	if (line.substr(0, 3) != "Kd=") {
		KOBRA_LOG_FUNC(error) << "Expected Kd= but got " << line << std::endl;
		return std::move(mat);
	}

	std::sscanf(line.c_str(), "Kd=%f %f %f", &Kd.x, &Kd.y, &Kd.z);

	// Read Ks
	glm::vec3 Ks;
	std::getline(file, line);

	// Ensure correct header ("Ks=")
	if (line.substr(0, 3) != "Ks=") {
		KOBRA_LOG_FUNC(error) << "Expected Ks= but got " << line << std::endl;
		return std::move(mat);
	}

	std::sscanf(line.c_str(), "Ks=%f %f %f", &Ks.x, &Ks.y, &Ks.z);

	// Read shading type
	std::getline(file, line);

	// line = "shading_type=..."
	// TODO: ensure the header is correct
	std::string stype = line.substr(13);
	std::optional <Shading> type = shading_from_str(stype);
	KOBRA_ASSERT(type.has_value(), "Invalid shading type \"" + stype + "\"");

	Shading shading = type.value();

	// Read refr_eta
	float refr_eta;
	std::getline(file, line);

	// Ensure correct header ("index_of_refraction=")
	if (line.substr(0, 20) != "index_of_refraction=") {
		KOBRA_LOG_FUNC(error) << "Expected index_of_refraction= but got "
			<< line.substr(0, 20) << std::endl;
		return std::move(mat);
	}

	std::sscanf(line.c_str(), "index_of_refraction=%f", &refr_eta);

	// Read refr_k
	float refr_k;
	std::getline(file, line);

	// Ensure correct header ("extinction_coefficient=")
	if (line.substr(0, 23) != "extinction_coefficient=") {
		KOBRA_LOG_FUNC(error) << "Expected extinction_coefficient= but got " << line << std::endl;
		return std::move(mat);
	}

	std::sscanf(line.c_str(), "extinction_coefficient=%f", &refr_k);

	// Read albedo source
	char buf2[1024];
	std::string albedo_texture;
	std::getline(file, line);

	// Ensure correct header ("albedo_texture=")
	if (line.substr(0, 15) != "albedo_texture=") {
		KOBRA_LOG_FUNC(error) << "Expected albedo_texture= but got " << line << std::endl;
		return std::move(mat);
	}

	std::sscanf(line.c_str(), "albedo_texture=%s", buf2);
	albedo_texture = buf2;

	// Read normal source
	char buf3[1024];
	std::string normal_texture;
	std::getline(file, line);

	// Ensure correct header ("normal_texture=")
	if (line.substr(0, 15) != "normal_texture=") {
		KOBRA_LOG_FUNC(error) << "Expected normal_texture= but got " << line << std::endl;
		return std::move(mat);
	}

	std::sscanf(line.c_str(), "normal_texture=%s", buf3);
	normal_texture = buf3;

	mat.diffuse = Kd;
	mat.specular = Ks;

	mat.type = shading;

	mat.refraction = refr_eta;
	// mat.refr_k = refr_k;

	// The actual image data isnt loaded until absolutely necessary

	// TODO: create a texture loader agent to handle multithreaded textures
	std::thread *albdeo_loader = nullptr;

	if (albedo_texture != "0") {
		// TODO: multithreading
		Profiler::one().frame("Loading albedo texture");
		albedo_texture = common::get_path(
			albedo_texture,
			common::get_directory(scene_file)
		);

		mat.albedo_texture = albedo_texture;
		// mat.set_albedo(phdev, device, command_pool, albedo_texture);
		Profiler::one().end();
	}

	if (normal_texture != "0") {
		Profiler::one().frame("Loading normal texture");
		normal_texture = common::get_path(
			normal_texture,
			common::get_directory(scene_file)
		);

		mat.normal_texture = normal_texture;
		// mat.set_normal(phdev, device, command_pool, normal_texture);
		Profiler::one().end();
	}

	/* Wait for albedo and normal to load
	if (albdeo_loader != nullptr) {
		albdeo_loader->join();
		delete albdeo_loader;
	} */

	// Return material
	success = true;
	return mat;
}

}
