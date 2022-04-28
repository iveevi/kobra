// Standard headers
#include <thread>

// Engine headers
#include "../include/material.hpp"
#include "../include/profiler.hpp"

namespace kobra {

// Copy constructor
Material::Material(const Material &other)
		: Kd(other.Kd), Ks(other.Ks),
		type(other.type),
		refr_eta(other.refr_eta),
		refr_k(other.refr_k)
{
	if (other.albedo_sampler != nullptr) {
		albedo_sampler = new Sampler(*other.albedo_sampler);
		albedo_source = other.albedo_source;
	}

	if (other.normal_sampler != nullptr) {
		normal_sampler = new Sampler(*other.normal_sampler);
		normal_source = other.normal_source;
	}
}

// Assignment operator
Material &Material::operator=(const Material &other)
{
	if (this != &other) {
		Kd = other.Kd;
		Ks = other.Ks;
		type = other.type;
		refr_eta = other.refr_eta;
		refr_k = other.refr_k;

		if (other.albedo_sampler != nullptr) {
			if (albedo_sampler != nullptr)
				delete albedo_sampler;

			albedo_sampler = new Sampler(*other.albedo_sampler);
			albedo_source = other.albedo_source;
		}

		if (other.normal_sampler != nullptr) {
			if (normal_sampler != nullptr)
				delete normal_sampler;

			normal_sampler = new Sampler(*other.normal_sampler);
			normal_source = other.normal_source;
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
	int i_type = static_cast <int> (type);
	float f_type = *reinterpret_cast <float *> (&i_type);

	bf_mats->push_back(aligned_vec4(Kd, f_type));
	bf_mats->push_back(aligned_vec4(
		{refr_eta, (!albedo_sampler), (!normal_sampler), 0}
	));
}

// Save material to file
void Material::save(std::ofstream &file) const
{
	file << "[MATERIAL]\n";
	file << "Kd=" << Kd.x << " " << Kd.y << " " << Kd.z << std::endl;
	file << "Ks=" << Ks.x << " " << Ks.y << " " << Ks.z << std::endl;
	file << "shading_type=" << shading_str(type) << std::endl;
	file << "index_of_refraction=" << refr_eta << std::endl;
	file << "extinction_coefficient=" << refr_k << std::endl;

	file << "albedo_texture=" << (albedo_sampler ? albedo_source : "0") << std::endl;
	file << "normal_texture=" << (normal_sampler ? normal_source : "0") << std::endl;
}

// Read material from file
std::optional <Material> Material::from_file
		(const Vulkan::Context &ctx,
		const VkCommandPool &command_pool,
		std::ifstream &file,
		const std::string &scene_file)
{
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
		return std::nullopt;
	}

	std::sscanf(line.c_str(), "Kd=%f %f %f", &Kd.x, &Kd.y, &Kd.z);

	// Read Ks
	glm::vec3 Ks;
	std::getline(file, line);

	// Ensure correct header ("Ks=")
	if (line.substr(0, 3) != "Ks=") {
		KOBRA_LOG_FUNC(error) << "Expected Ks= but got " << line << std::endl;
		return std::nullopt;
	}

	std::sscanf(line.c_str(), "Ks=%f %f %f", &Ks.x, &Ks.y, &Ks.z);

	std::cout << "Kd: " << Kd.x << " " << Kd.y << " " << Kd.z << std::endl;
	std::cout << "Ks: " << Ks.x << " " << Ks.y << " " << Ks.z << std::endl;

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
		return std::nullopt;
	}

	std::sscanf(line.c_str(), "index_of_refraction=%f", &refr_eta);

	// Read refr_k
	float refr_k;
	std::getline(file, line);

	// Ensure correct header ("extinction_coefficient=")
	if (line.substr(0, 23) != "extinction_coefficient=") {
		KOBRA_LOG_FUNC(error) << "Expected extinction_coefficient= but got " << line << std::endl;
		return std::nullopt;
	}

	std::sscanf(line.c_str(), "extinction_coefficient=%f", &refr_k);

	// Read albedo source
	char buf2[1024];
	std::string albedo_source;
	std::getline(file, line);

	// Ensure correct header ("albedo_texture=")
	if (line.substr(0, 15) != "albedo_texture=") {
		KOBRA_LOG_FUNC(error) << "Expected albedo_texture= but got " << line << std::endl;
		return std::nullopt;
	}

	std::sscanf(line.c_str(), "albedo_texture=%s", buf2);
	albedo_source = buf2;

	// Read normal source
	char buf3[1024];
	std::string normal_source;
	std::getline(file, line);

	// Ensure correct header ("normal_texture=")
	if (line.substr(0, 15) != "normal_texture=") {
		KOBRA_LOG_FUNC(error) << "Expected normal_texture= but got " << line << std::endl;
		return std::nullopt;
	}

	std::sscanf(line.c_str(), "normal_texture=%s", buf3);
	normal_source = buf3;

	// Construct and return material
	Material mat;

	mat.Kd = Kd;
	mat.Ks = Ks;

	mat.type = shading;

	mat.refr_eta = refr_eta;
	mat.refr_k = refr_k;

	std::cout << "refr_eta = " << refr_eta << std::endl;
	std::cout << "refr_k = " << refr_k << std::endl;

	// TODO: create a texture loader agent to handle multithreaded textures
	std::thread *albdeo_loader = nullptr;
	if (albedo_source != "0") {
		std::cout << "Loading albedo texture: " << albedo_source << std::endl;
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
		std::cout << "Loading normal texture: " << normal_source << std::endl;
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

	std::cout << "Material loaded" << std::endl;

	// Return material
	return mat;
}

}
