// OptiX headers
#include <optix_device.h>
#include <optix_host.h>
#include <optix_stack_size.h>

// Engine headers
#include "../../include/camera.hpp"
#include "../../include/cuda/alloc.cuh"
#include "../../include/cuda/cast.cuh"
#include "../../include/cuda/color.cuh"
#include "../../include/cuda/interop.cuh"
#include "../../include/ecs.hpp"
#include "../../include/amadeus/armada.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/transform.hpp"
#include "../../shaders/raster/bindings.h"
#include "../../include/profiler.hpp"

namespace kobra {

namespace amadeus {

// Create the layer
// TODO: all custom extent...
ArmadaRTX::ArmadaRTX(const Context &context,
		const std::shared_ptr <amadeus::System> &system,
		const std::shared_ptr <layers::MeshMemory> &mesh_memory,
		const vk::Extent2D &extent)
		: m_system(system), m_mesh_memory(mesh_memory),
		m_device(context.device), m_phdev(context.phdev),
		m_texture_loader(context.texture_loader),
		m_extent(extent), m_active_attachment()
{
	// Start the timer
	m_timer.start();

	// Initialize the host state
	m_host.last_updated = 0;

	// Initialize TLAS state
	m_tlas.null = true;
	m_tlas.last_updated = 0;

	// Configure launch parameters
	auto &params = m_launch_info;

	params.resolution = {
		extent.width,
		extent.height
	};

	params.max_depth = 10;
	params.samples = 0;
	params.accumulate = true;
	params.lights.quad_count = 0;
	params.lights.tri_count = 0;
	params.materials = nullptr;
	params.environment_map = 0;
	params.has_environment_map = false;

	// Allocate results
	int size = extent.width * extent.height;

	params.buffers.color = cuda::alloc <glm::vec4> (size);
	params.buffers.normal = cuda::alloc <glm::vec4> (size);
	params.buffers.albedo = cuda::alloc <glm::vec4> (size);
	params.buffers.position = cuda::alloc <glm::vec4> (size);

	// Add self to the material system ping list
	Material::daemon.ping_at(this,
		[](void *user, const std::set <uint32_t> &materials) {
			ArmadaRTX *armada = (ArmadaRTX *) user;
			armada->update_materials(materials);
		}
	);
}

// Set the environment map
void ArmadaRTX::set_envmap(const std::string &path)
{
	// First load the environment map
	const auto &map = m_texture_loader->load_texture(path);
	m_launch_info.environment_map = cuda::import_vulkan_texture(*m_device, map);
	m_launch_info.has_environment_map = true;
}

// Update the light buffers if needed
void ArmadaRTX::update_light_buffers
		(const std::vector <const Light *> &lights,
		const std::vector <const Transform *> &light_transforms,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
{
	// TODO: lighting system equivalent of System
	if (m_host.quad_lights.size() != lights.size()) {
		m_host.quad_lights.resize(lights.size());

		auto &quad_lights = m_host.quad_lights;
		for (int i = 0; i < lights.size(); i++) {
			const Light *light = lights[i];
			const Transform *transform = light_transforms[i];

			glm::vec3 a {-0.5f, 0, -0.5f};
			glm::vec3 b {0.5f, 0, -0.5f};
			glm::vec3 c {-0.5f, 0, 0.5f};

			a = transform->apply(a);
			b = transform->apply(b);
			c = transform->apply(c);

			quad_lights[i].a = cuda::to_f3(a);
			quad_lights[i].ab = cuda::to_f3(b - a);
			quad_lights[i].ac = cuda::to_f3(c - a);
			quad_lights[i].intensity = cuda::to_f3(light->power * light->color);
		}

		m_launch_info.lights.quad_lights = cuda::make_buffer(quad_lights);
		m_launch_info.lights.quad_count = quad_lights.size();

		KOBRA_LOG_FUNC(Log::INFO) << "Uploaded " << quad_lights.size()
			<< " quad lights to the GPU\n";
	}

	// Count number of emissive submeshes
	int emissive_count = 0;

	// TODO: compute before hand
	std::vector <std::pair <const Submesh *, int>> emissive_submeshes;
	for (int i = 0; i < submeshes.size(); i++) {
		const Submesh *submesh = submeshes[i];
		const Material &material = Material::all[submesh->material_index];
		if (glm::length(material.emission) > 0
				|| material.has_emission()) {
			emissive_submeshes.push_back({submesh, i});
			emissive_count += submesh->triangles();
		}
	}

	if (m_host.tri_lights.size() != emissive_count) {
		for (const auto &pr : emissive_submeshes) {
			const Submesh *submesh = pr.first;
			const Transform *transform = submesh_transforms[pr.second];
			const Material &material = Material::all[submesh->material_index];

			for (int i = 0; i < submesh->triangles(); i++) {
				uint32_t i0 = submesh->indices[i * 3 + 0];
				uint32_t i1 = submesh->indices[i * 3 + 1];
				uint32_t i2 = submesh->indices[i * 3 + 2];

				glm::vec3 a = transform->apply(submesh->vertices[i0].position);
				glm::vec3 b = transform->apply(submesh->vertices[i1].position);
				glm::vec3 c = transform->apply(submesh->vertices[i2].position);

				m_host.tri_lights.push_back(
					optix::TriangleLight {
						cuda::to_f3(a),
						cuda::to_f3(b - a),
						cuda::to_f3(c - a),
						cuda::to_f3(material.emission)
						// TODO: what if material has
						// textured emission?
					}
				);
			}
		}

		m_launch_info.lights.tri_lights = cuda::make_buffer(m_host.tri_lights);
		m_launch_info.lights.tri_count = m_host.tri_lights.size();

		// TODO: display logging in UI as well (add log routing)
		KOBRA_LOG_FUNC(Log::INFO) << "Uploaded " << m_host.tri_lights.size()
			<< " triangle lights to the GPU\n";
	}
}

// Update the SBT data
void ArmadaRTX::update_sbt_data
		(const std::vector <layers::MeshMemory::Cachelet> &cachelets,
		const std::vector <const Submesh *> &submeshes,
		const std::vector <const Transform *> &submesh_transforms)
{
	int submesh_count = submeshes.size();

	m_host.hit_records.clear();
	for (int i = 0; i < submesh_count; i++) {
		const Submesh *submesh = submeshes[i];
		const Material &mat = Material::all[submesh->material_index];

		HitRecord hit_record {};

		hit_record.data.model = submesh_transforms[i]->matrix();
		hit_record.data.material_index = submesh->material_index;

		hit_record.data.triangles = cachelets[i].m_cuda_triangles;
		hit_record.data.vertices = cachelets[i].m_cuda_vertices;

		// Push back
		m_host.hit_records.push_back(hit_record);
	}
}

void ArmadaRTX::update_materials(const std::set <uint32_t> &material_indices)
{
	for (uint32_t mat_index : material_indices) {
		const Material &material = Material::all[mat_index];
		cuda::_material &mat = m_host.materials[mat_index];

		// Copy basic data
		mat.diffuse = cuda::to_f3(material.diffuse);
		mat.specular = cuda::to_f3(material.specular);
		mat.emission = cuda::to_f3(material.emission);
		mat.ambient = cuda::to_f3(material.ambient);
		mat.shininess = material.shininess;
		mat.roughness = material.roughness;
		mat.refraction = material.refraction;
		mat.type = material.type;

		// TODO: textures
	}

	// Copy to GPU
	// TODO: only copy subregions
	cudaMemcpy(m_launch_info.materials,
		m_host.materials.data(),
		m_host.materials.size() * sizeof(cuda::_material),
		cudaMemcpyHostToDevice
	);
}

// Preprocess scene data
// TODO: get rid of this method..
ArmadaRTX::preprocess_update ArmadaRTX::preprocess_scene
		(const ECS &ecs,
		const Camera &camera,
		const Transform &transform)
{
	// To return
	std::optional <OptixTraversableHandle> handle;
	std::vector <HitRecord> *hit_records = nullptr;

	// Set viewing position
	m_launch_info.camera.center = transform.position;

	auto uvw = kobra::uvw_frame(camera, transform);

	m_launch_info.camera.ax_u = uvw.u;
	m_launch_info.camera.ax_v = uvw.v;
	m_launch_info.camera.ax_w = uvw.w;

	m_launch_info.camera.projection = camera.perspective_matrix();
	m_launch_info.camera.view = camera.view_matrix(transform);

	// Get time
	m_launch_info.time = m_timer.elapsed_start();

	// Update the raytracing system
	bool updated = m_system->update(ecs);

	// Preprocess the entities
	std::vector <const Renderable *> rasterizers;
	std::vector <const Transform *> rasterizer_transforms;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	for (int i = 0; i < ecs.size(); i++) {
		// TODO: one unifying renderer component, with options for
		// raytracing, etc
		if (ecs.exists <Renderable> (i)) {
			const auto *rasterizer = &ecs.get <Renderable> (i);
			const auto *transform = &ecs.get <Transform> (i);

			rasterizers.push_back(rasterizer);
			rasterizer_transforms.push_back(transform);
		}

		if (ecs.exists <Light> (i)) {
			const auto *light = &ecs.get <Light> (i);
			const auto *transform = &ecs.get <Transform> (i);

			lights.push_back(light);
			light_transforms.push_back(transform);
		}
	}

	// Update data if necessary
	if (updated || m_tlas.null) {
		// Load the list of all submeshes
		std::vector <layers::MeshMemory::Cachelet> cachelets; // TODO: redo this method...
		std::vector <const Submesh *> submeshes;
		std::vector <const Transform *> submesh_transforms;

		for (int i = 0; i < rasterizers.size(); i++) {
			const Renderable *rasterizer = rasterizers[i];
			const Transform *transform = rasterizer_transforms[i];

			// Cache the renderables
			// TODO: all update functions should go to a separate methods
			m_mesh_memory->cache_cuda(rasterizer);

			for (int j = 0; j < rasterizer->mesh->submeshes.size(); j++) {
				const Submesh *submesh = &rasterizer->mesh->submeshes[j];

				cachelets.push_back(m_mesh_memory->get(rasterizer, j));
				submeshes.push_back(submesh);
				submesh_transforms.push_back(transform);
			}
		}

		// Update the data
		update_light_buffers(
			lights, light_transforms,
			submeshes, submesh_transforms
		);

		update_sbt_data(cachelets, submeshes, submesh_transforms);
		// hit_records = &m_host.hit_records;
		m_host.last_updated = clock();

		// Reset the number of samples stored
		m_launch_info.samples = 0;

		// Update TLAS state
		m_tlas.null = false;
		m_tlas.last_updated = clock();

		// Update the status
		updated = true;
	}

	// Generate material buffer if needed
	if (!m_launch_info.materials) {
		std::cout << "Generating material buffer" << std::endl;

		m_host.materials.clear();
		for (const Material &material : Material::all) {
			cuda::_material mat;

			// Scalar/vector values
			mat.diffuse = cuda::to_f3(material.diffuse);
			mat.specular = cuda::to_f3(material.specular);
			mat.emission = cuda::to_f3(material.emission);
			mat.ambient = cuda::to_f3(material.ambient);
			mat.shininess = material.shininess;
			mat.roughness = material.roughness;
			mat.refraction = material.refraction;
			mat.type = material.type;

			// Textures
			if (material.has_albedo()) {
				const ImageData &diffuse = m_texture_loader
					->load_texture(material.albedo_texture);

				mat.textures.diffuse
					= cuda::import_vulkan_texture(*m_device, diffuse);
				mat.textures.has_diffuse = true;
			}

			if (material.has_normal()) {
				const ImageData &normal = m_texture_loader
					->load_texture(material.normal_texture);

				mat.textures.normal
					= cuda::import_vulkan_texture(*m_device, normal);
				mat.textures.has_normal = true;
			}

			if (material.has_specular()) {
				const ImageData &specular = m_texture_loader
					->load_texture(material.specular_texture);

				mat.textures.specular
					= cuda::import_vulkan_texture(*m_device, specular);
				mat.textures.has_specular = true;
			}

			if (material.has_emission()) {
				const ImageData &emission = m_texture_loader
					->load_texture(material.emission_texture);

				mat.textures.emission
					= cuda::import_vulkan_texture(*m_device, emission);
				mat.textures.has_emission = true;
			}

			if (material.has_roughness()) {
				const ImageData &roughness = m_texture_loader
					->load_texture(material.roughness_texture);

				mat.textures.roughness
					= cuda::import_vulkan_texture(*m_device, roughness);
				mat.textures.has_roughness = true;
			}

			m_host.materials.push_back(mat);
		}

		m_launch_info.materials = cuda::make_buffer(m_host.materials);
	}

	// Send hit records to attachment if needed
	long long int attachment_time = m_host.times[m_previous_attachment];
	if (attachment_time < m_host.last_updated) {
		// Send the hit records
		hit_records = &m_host.hit_records;
		m_host.times[m_previous_attachment] = m_host.last_updated;
	}

	// Create acceleration structure for the attachment if needed
	// assuming that there is currently a valid attachment
	attachment_time = m_tlas.times[m_previous_attachment];
	if (attachment_time < m_tlas.last_updated) {
		// Create the acceleration structure
		m_tlas.times[m_previous_attachment] = m_tlas.last_updated;
		handle = m_system->build_tlas(
			rasterizers,
			m_attachments[m_previous_attachment]->m_hit_group_count
		);
	}

	return {handle, hit_records};
}

// Path tracing computation
void ArmadaRTX::render(const ECS &ecs,
		const Camera &camera,
		const Transform &transform,
		bool accumulate)
{
	// Skip and warn if no active attachment
	if (m_active_attachment.empty()) {
		KOBRA_LOG_FUNC(Log::WARN) << "No active attachment\n";
		return;
	}

	// Compare with previous attachment
	if (m_active_attachment != m_previous_attachment) {
		if (m_previous_attachment.size() > 0)
			m_attachments[m_previous_attachment]->unload();

		m_previous_attachment = m_active_attachment;
		m_attachments[m_previous_attachment]->load();
	}

	auto out = preprocess_scene(ecs, camera, transform);

	// Reset the accumulation state if needed
	if (!accumulate)
		m_launch_info.samples = 0;

	// Invoke render for current attachment
	auto &attachment = m_attachments[m_previous_attachment];
	attachment->render(this, m_launch_info, out.handle, out.hit_records, m_extent);

	// Increment number of samples
	m_launch_info.samples++;
}

}

}
