// Standard headers
#include <thread>

// Engine headers
#include "../include/material.hpp"
#include "../include/profiler.hpp"

namespace kobra {

// Constructors
Material::Material(const glm::vec3 &albedo,
		float shading_type,
		float ior)
		: albedo(albedo), shading_type(shading_type),
		ior(ior) {}

// Copy constructor
Material::Material(const Material &other)
		: albedo(other.albedo), shading_type(other.shading_type),
		ior(other.ior)
{
	if (other.albedo_sampler != nullptr)
		albedo_sampler = new Sampler(*other.albedo_sampler);

	if (other.normal_sampler != nullptr)
		normal_sampler = new Sampler(*other.normal_sampler);
}

// Assignment operator
Material &Material::operator=(const Material &other)
{
	if (this != &other) {
		albedo = other.albedo;
		shading_type = other.shading_type;
		ior = other.ior;

		if (other.albedo_sampler != nullptr) {
			if (albedo_sampler != nullptr)
				delete albedo_sampler;

			albedo_sampler = new Sampler(*other.albedo_sampler);
		}

		if (other.normal_sampler != nullptr) {
			if (normal_sampler != nullptr)
				delete normal_sampler;

			normal_sampler = new Sampler(*other.normal_sampler);
		}
	}

	return *this;
}

// Destructor
Material::~Material()
{
	if (albedo_sampler != nullptr)
		delete albedo_sampler;

	if (normal_sampler != nullptr)
		delete normal_sampler;
}

// Properties
bool Material::has_albedo() const
{
	return albedo_sampler != nullptr;
}

bool Material::has_normal() const
{
	return normal_sampler != nullptr;
}

// Get image descriptors
std::optional <VkDescriptorImageInfo> Material::get_albedo_descriptor() const
{
	if (albedo_sampler == nullptr)
		return std::nullopt;
	return albedo_sampler->get_image_info();	// TODO: refactor
}

std::optional <VkDescriptorImageInfo> Material::get_normal_descriptor() const
{
	if (normal_sampler == nullptr)
		return std::nullopt;
	return normal_sampler->get_image_info();	// TODO: refactor
}

// Bind albdeo and normal textures
void Material::bind(const Vulkan::Context &context,
		const VkCommandPool &command_pool,
		const VkDescriptorSet &ds, size_t b1, size_t b2) const
{
	// Blank sampler
	Sampler blank = Sampler::blank_sampler(context, command_pool);

	// Bind albedo
	if (albedo_sampler != nullptr)
		albedo_sampler->bind(ds, b1);
	else
		blank.bind(ds, b1);

	// Bind normal
	if (normal_sampler != nullptr)
		normal_sampler->bind(ds, b2);
	else
		blank.bind(ds, b2);
}

// Set textures
void Material::set_albedo(const Vulkan::Context &ctx,
		const VkCommandPool &command_pool,
		const std::string &path)
{
	albedo_source = path;
	albedo_sampler = new Sampler(ctx, command_pool, path);
}

void Material::set_normal(const Vulkan::Context &ctx,
		const VkCommandPool &command_pool,
		const std::string &path)
{
	normal_source = path;
	normal_sampler = new Sampler(ctx, command_pool, path);
}

// Serialize to GPU buffer
void Material::serialize(Buffer4f *bf_mats) const
{
	bf_mats->push_back(aligned_vec4(albedo, shading_type));
	bf_mats->push_back(aligned_vec4(
		{ior, (!albedo_sampler), (!normal_sampler), 0}
	));
}

// Save material to file
void Material::save(std::ofstream &file) const
{
	char buf[1024];
	sprintf(buf, "%d", *((int *) &shading_type));

	file << "[MATERIAL]\n";
	file << "albedo=" << albedo.x << " " << albedo.y << " " << albedo.z << std::endl;
	file << "shading_type=" << buf << std::endl;
	file << "ior=" << ior << std::endl;

	file << "albedo_source=" << (albedo_sampler ? albedo_source : "0") << std::endl;
	file << "normal_source=" << (normal_sampler ? normal_source : "0") << std::endl;
}

// Read material from file
std::optional <Material> Material::from_file
		(const Vulkan::Context &ctx,
		const VkCommandPool &command_pool,
		std::ifstream &file,
		const std::string &scene_file)
{
	std::string line;

	// Read albedo
	glm::vec3 albedo;
	std::getline(file, line);
	std::sscanf(line.c_str(), "albedo=%f %f %f", &albedo.x, &albedo.y, &albedo.z);

	// Read shading type
	char buf1[1024];
	float shading_type;
	int tmp;
	std::getline(file, line);
	std::sscanf(line.c_str(), "shading_type=%s", buf1);

	// TODO: helper method
	// Check if the shading type is literal
	std::string stype = buf1;
	if (stype == "DIFFUSE") {
		shading_type = SHADING_TYPE_DIFFUSE;
	} else if (stype == "EMISSIVE") {
		shading_type = SHADING_TYPE_EMISSIVE;
	} else if (stype == "REFLECTION") {
		shading_type = SHADING_TYPE_REFLECTION;
	} else if (stype == "REFRACTION") {
		shading_type = SHADING_TYPE_REFRACTION;
	} else {
		std::sscanf(buf1, "%d", &tmp);
		shading_type = tmp;
		shading_type = *((float *) &tmp);
	}

	// Read ior
	float ior;
	std::getline(file, line);
	std::sscanf(line.c_str(), "ior=%f", &ior);

	// Read albedo source
	char buf2[1024];
	std::string albedo_source;
	std::getline(file, line);
	std::sscanf(line.c_str(), "albedo_source=%s", buf2);
	albedo_source = buf2;

	// Read normal source
	char buf3[1024];
	std::string normal_source;
	std::getline(file, line);
	std::sscanf(line.c_str(), "normal_source=%s", buf3);
	normal_source = buf3;

	// Construct and return material
	Material mat;
	mat.albedo = albedo;
	mat.shading_type = shading_type;
	mat.ior = ior;

	// TODO: create a texture loader agent to handle multithreaded textures
	std::thread *albdeo_loader = nullptr;
	if (albedo_source != "0") {
		auto load_albedo = [&]() {
			Profiler::one().frame("Loading albedo texture");
			albedo_source = common::get_path(
				albedo_source,
				common::get_directory(scene_file)
			);

			mat.set_albedo(ctx, command_pool, albedo_source);
			Profiler::one().end();
		};

		albdeo_loader = new std::thread(load_albedo);
	}

	if (normal_source != "0") {
		Profiler::one().frame("Loading normal texture");
		normal_source = common::get_path(
			normal_source,
			common::get_directory(scene_file)
		);

		mat.set_normal(ctx, command_pool, normal_source);
		Profiler::one().end();
	}

	// Wait for albedo and normal to load
	if (albdeo_loader != nullptr) {
		albdeo_loader->join();
		delete albdeo_loader;
	}

	// Return material
	return mat;
}

}
