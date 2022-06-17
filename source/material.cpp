// Standard headers
#include <thread>

// Engine headers
#include "../include/material.hpp"
#include "../include/profiler.hpp"

namespace kobra {

// Properties
bool Material::has_albedo() const
{
	return !(albedo_source.empty() || albedo_source == "0");
}

bool Material::has_normal() const
{
	return !(normal_source.empty() || normal_source == "0");
}

// Get image descriptors
std::optional <vk::DescriptorImageInfo> Material::get_albedo_descriptor() const
{
	if (albedo_source.empty())
		return std::nullopt;

	return vk::DescriptorImageInfo {
		*albedo_sampler,
		*albedo_image.view,
		vk::ImageLayout::eShaderReadOnlyOptimal
	};
}

std::optional <vk::DescriptorImageInfo> Material::get_normal_descriptor() const
{
	if (normal_source.empty())
		return std::nullopt;

	return vk::DescriptorImageInfo {
		*normal_sampler,
		*normal_image.view,
		vk::ImageLayout::eShaderReadOnlyOptimal
	};
}

// Bind albdeo and normal textures
void Material::bind(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		const vk::raii::DescriptorSet &dset,
		uint32_t b1, uint32_t b2)
{
	// Bind albedo
	if (albedo_source.length() > 0) {
		bind_ds(device, dset, albedo_sampler, albedo_image, b1);
	} else {
		albedo_image = ImageData::blank(phdev, device);
		albedo_image.transition_layout(device, command_pool, vk::ImageLayout::eShaderReadOnlyOptimal);

		albedo_sampler = make_sampler(device, albedo_image);
		albedo_source = "0";
		bind_ds(device, dset, albedo_sampler, albedo_image, b1);
	}

	// Bind normal
	if (normal_source.length() > 0) {
		bind_ds(device, dset, normal_sampler, normal_image, b2);
	} else {
		normal_image = ImageData::blank(phdev, device);
		normal_image.transition_layout(device, command_pool, vk::ImageLayout::eShaderReadOnlyOptimal);

		normal_sampler = make_sampler(device, normal_image);
		normal_source = "0";
		bind_ds(device, dset, normal_sampler, normal_image, b2);
	}
}

// Set textures
void Material::set_albedo(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		const std::string &path)
{
	albedo_source = path;

	albedo_image = make_image(phdev, device,
		command_pool, path,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst
			| vk::ImageUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	albedo_sampler = make_sampler(device, albedo_image);
}

void Material::set_normal(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		const std::string &path)
{
	normal_source = path;

	normal_image = make_image(phdev, device,
		command_pool, path,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst
			| vk::ImageUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	normal_sampler = make_sampler(device, normal_image);
}

// Serialize to buffer
void Material::serialize(std::vector <aligned_vec4> &buffer) const
{
	int i_type = static_cast <int> (type);
	float f_type = *reinterpret_cast <float *> (&i_type);

	buffer.push_back(aligned_vec4(Kd, f_type));
	buffer.push_back(aligned_vec4(
		{refr_eta, albedo_source.empty(), normal_source.empty(), 0}
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

	file << "albedo_texture=" << (albedo_source.empty() ? "0" : albedo_source) << std::endl;
	file << "normal_texture=" << (normal_source.empty() ? "0" : normal_source) << std::endl;
}

// Read material from file
Material Material::from_file
		(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		std::ifstream &file,
		const std::string &scene_file,
		bool &success)
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
	std::string albedo_source;
	std::getline(file, line);

	// Ensure correct header ("albedo_texture=")
	if (line.substr(0, 15) != "albedo_texture=") {
		KOBRA_LOG_FUNC(error) << "Expected albedo_texture= but got " << line << std::endl;
		return std::move(mat);
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
		return std::move(mat);
	}

	std::sscanf(line.c_str(), "normal_texture=%s", buf3);
	normal_source = buf3;

	mat.Kd = Kd;
	mat.Ks = Ks;

	mat.type = shading;

	mat.refr_eta = refr_eta;
	mat.refr_k = refr_k;

	std::cout << "refr_eta = " << refr_eta << std::endl;
	std::cout << "refr_k = " << refr_k << std::endl;

	// TODO: create a texture loader agent to handle multithreaded textures
	std::thread *albdeo_loader = nullptr;

	// Create new command_pool
	vk::raii::CommandPool cmd_pool {
		device, {
			vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				find_graphics_queue_family(phdev)
		}
	};

	if (albedo_source != "0") {
		std::cout << "Loading albedo texture: " << albedo_source << std::endl;

		auto load_albedo = [&]() {
			Profiler::one().frame("Loading albedo texture");
			albedo_source = common::get_path(
				albedo_source,
				common::get_directory(scene_file)
			);

			mat.set_albedo(phdev, device, cmd_pool, albedo_source);
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

		mat.set_normal(phdev, device, command_pool, normal_source);
		Profiler::one().end();
	}

	// Wait for albedo and normal to load
	if (albdeo_loader != nullptr) {
		albdeo_loader->join();
		delete albdeo_loader;
	}

	std::cout << "Material loaded" << std::endl;

	// Return material
	success = true;
	return mat; 
}

}
