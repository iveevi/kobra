// OptiX headers
#include <optix_device.h>
#include <optix_host.h>
#include <optix_stack_size.h>

// ImGUI headers
#include <imgui.h>

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
	params.lights.quad_lights = nullptr;
	params.lights.quad_count = 0;
	params.lights.tri_lights = nullptr;
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

void ArmadaRTX::update_triangle_light_buffers
		(const std::set <_instance_ref> &emissive_submeshes_to_update)
{
	// TODO: share this setup with the renderables (another layer for
	// material buffer updates? or use the same daemon?)
	if (m_host.tri_lights.size() != m_host.emissive_count) {
		if (m_launch_info.lights.tri_lights) {
			// TODO: free this buffer only when rendering is
			// complete...
			cuda::free(m_launch_info.lights.tri_lights);
			m_launch_info.lights.tri_lights = nullptr;
			m_launch_info.lights.tri_count = 0;
		}

		m_host.tri_lights.clear();
		m_host.emissive_submesh_offsets.clear();

		if (m_host.emissive_count <= 0)
			return;

		for (const auto &pr : m_host.emissive_submeshes) {
			const Submesh *submesh = pr.submesh;
			const Transform *transform = pr.transform;

			const Material &material = Material::all[submesh->material_index];

			m_host.emissive_submesh_offsets[submesh] = m_host.tri_lights.size();
			for (int i = 0; i < submesh->triangles(); i++) {
				uint32_t i0 = submesh->indices[i * 3 + 0];
				uint32_t i1 = submesh->indices[i * 3 + 1];
				uint32_t i2 = submesh->indices[i * 3 + 2];

				glm::vec3 a = transform->apply(submesh->vertices[i0].position);
				glm::vec3 b = transform->apply(submesh->vertices[i1].position);
				glm::vec3 c = transform->apply(submesh->vertices[i2].position);

				// TODO: cache cuda textures
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

		std::cout << "# of triangle lights: " << m_host.tri_lights.size() << "\n";

		m_launch_info.lights.tri_lights = cuda::make_buffer(m_host.tri_lights);
		m_launch_info.lights.tri_count = m_host.tri_lights.size();

		// TODO: display logging in UI as well (add log routing)
		KOBRA_LOG_FUNC(Log::INFO) << "Uploaded " << m_host.tri_lights.size()
			<< " triangle lights to the GPU\n";
	} else if (emissive_submeshes_to_update.size() > 0) {
		size_t subrange_min = std::numeric_limits <size_t>::max();
		size_t subrange_max = 0;

		for (const auto &pr : emissive_submeshes_to_update) {
			const Submesh *submesh = pr.submesh;
			const Transform *transform = pr.transform;

			const Material &material = Material::all[submesh->material_index];

			size_t offset = m_host.emissive_submesh_offsets[submesh];
			for (int i = 0; i < submesh->triangles(); i++) {
				// TODO: check if transforms have changed; if so
				// then also update the triangle light position
				m_host.tri_lights[offset + i].intensity =
					cuda::to_f3(material.emission);
			}

			subrange_min = std::min(subrange_min, offset);
			subrange_max = std::max(subrange_max, offset + submesh->triangles());
		}

		cudaMemcpy(
			&m_launch_info.lights.tri_lights[subrange_min],
			&m_host.tri_lights[subrange_min],
			(subrange_max - subrange_min) * sizeof(optix::TriangleLight),
			cudaMemcpyHostToDevice
		);
	}
}

// Update the light buffers if needed
void ArmadaRTX::update_quad_light_buffers
		(const std::vector <const Light *> &lights,
		const std::vector <const Transform *> &light_transforms)
{
	// TODO: lighting system equivalent of System
	if (m_host.quad_lights.size() != lights.size()) {
		if (m_launch_info.lights.quad_lights)
			cuda::free(m_launch_info.lights.quad_lights);

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

                // TODO: use ecs indirection...
		hit_record.data.model = submesh_transforms[i]->matrix();
		hit_record.data.material_index = submesh->material_index;

		hit_record.data.triangles = cachelets[i].m_cuda_triangles;
		hit_record.data.vertices = cachelets[i].m_cuda_vertices;

		// If the material is emissive, then we need to
		//	give a valid light index
		hit_record.data.light_index = -1;
		if (glm::length(mat.emission) > 0.0f) {
			hit_record.data.light_index =
				m_host.emissive_submesh_offsets[submesh];
		}

		// Push back
		m_host.hit_records.push_back(hit_record);
	}
}

void ArmadaRTX::update_materials(const std::set <uint32_t> &material_indices)
{
	// If host buffer is empty, assume the armada is not initialized
	if (m_host.materials.size() == 0)
		return;

	std::set <_instance_ref> emissive_submeshes_to_update;
	for (uint32_t mat_index : material_indices) {
		const Material &material = Material::all[mat_index];
		cuda::_material &mat = m_host.materials[mat_index];

		bool was_emissive = (length(mat.emission) > 0.0f)
				|| mat.textures.has_emission;

		// Copy basic data
		mat.diffuse = cuda::to_f3(material.diffuse);
		mat.specular = cuda::to_f3(material.specular);
		mat.emission = cuda::to_f3(material.emission);
		mat.ambient = cuda::to_f3(material.ambient);
		mat.shininess = material.shininess;
		mat.roughness = material.roughness;
		mat.refraction = material.refraction;
		mat.type = material.type;

		bool is_emissive = (length(mat.emission) > 0.0f)
				|| mat.textures.has_emission;

		// TODO: textures

		const auto &refs = m_host.material_submeshes[mat_index];

		// TODO: check previous state (to see whether to remove from
		// emissive submeshes)
		if (is_emissive) {
			for (const auto &pr : refs) {
				emissive_submeshes_to_update.insert(pr);
				if (m_host.emissive_submeshes.find(pr) !=
						m_host.emissive_submeshes.end())
					continue;

				m_host.emissive_submeshes.insert(pr);
				m_host.emissive_count += pr.submesh->triangles();
			}
		} else if (was_emissive && !is_emissive) {
			// Remove from emissive submeshes
			for (const auto &pr : refs) {
				m_host.emissive_submeshes.erase(pr);
				m_host.emissive_count -= pr.submesh->triangles();
			}
		}

		// TODO: what if the net change is 0 (with multiple material
		// emission changes)?
	}

	// Copy to GPU
	// TODO: only copy subregions
	cudaMemcpy(m_launch_info.materials,
		m_host.materials.data(),
		m_host.materials.size() * sizeof(cuda::_material),
		cudaMemcpyHostToDevice
	);

	bool sbt_needs_update = (m_host.tri_lights.size() != m_host.emissive_count);

	// Also update the emissive submeshes if needed
	// TODO: use the reutrn from tihs instead to check for sbt udpate...
	update_triangle_light_buffers(emissive_submeshes_to_update);

	// Update the SBT if needed (e.g. when a new emissive submesh is added)
	if (sbt_needs_update) {
		update_sbt_data(
			m_host.cachelets,
			m_host.submeshes,
			m_host.submesh_transforms
		);

		m_host.last_updated = clock();
	}
}

// Preprocess scene data
// TODO: get rid of this method..
ArmadaRTX::preprocess_update ArmadaRTX::preprocess_scene
		(const ECS &ecs,
                const daemons::Transform &transform_daemon,
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
        // TODO: helper method for this... (tuples)
        std::vector <int> renderable_id;
	std::vector <const Renderable *> renderables;
	std::vector <const Transform *> renderable_transforms;

	std::vector <const Light *> lights;
	std::vector <const Transform *> light_transforms;

	for (int i = 0; i < ecs.size(); i++) {
		// TODO: one unifying renderer component, with options for
		// raytracing, etc
		if (ecs.exists <Renderable> (i)) {
			const auto *renderable = &ecs.get <Renderable> (i);
			const auto *transform = &ecs.get <Transform> (i);

                        renderable_id.push_back(i);
			renderables.push_back(renderable);
			renderable_transforms.push_back(transform);
		}

		if (ecs.exists <Light> (i)) {
			const auto *light = &ecs.get <Light> (i);
			const auto *transform = &ecs.get <Transform> (i);

			lights.push_back(light);
			light_transforms.push_back(transform);
		}
	}

	// Update data if necessary
	if (m_tlas.null) {
		/* Load the list of all submeshes
		std::vector <layers::MeshMemory::Cachelet> cachelets; // TODO: redo this method...
		std::vector <const Submesh *> submeshes;
		std::vector <const Transform *> submesh_transforms; */

		m_host.cachelets.clear();
		m_host.submesh_transforms.clear();
		m_host.submeshes.clear();

		// Reserve material-submesh reference structure
		m_host.material_submeshes.clear();
		m_host.material_submeshes.resize(Material::all.size());

		for (int i = 0; i < renderables.size(); i++) {
                        int id = renderable_id[i];
			const Renderable *renderable = renderables[i];
			const Transform *transform = renderable_transforms[i];

			// Cache the renderables
			// TODO: all update functions should go to a separate methods
			m_mesh_memory->cache_cuda(renderable);

			for (int j = 0; j < renderable->mesh->submeshes.size(); j++) {
				const Submesh *submesh = &renderable->mesh->submeshes[j];
				uint32_t material_index = submesh->material_index;
				m_host.material_submeshes[material_index].insert(
					{transform, submesh, id}
				);

				m_host.cachelets.push_back(m_mesh_memory->get(renderable, j));

                                // TODO: use instance ref vector instead...
                                m_host.entity_id.push_back(i);
				m_host.submeshes.push_back(submesh);
				m_host.submesh_transforms.push_back(transform);
			}
		}

		// Count number of emissive submeshes
		m_host.emissive_count = 0;

		// TODO: compute before hand
		for (int i = 0; i < m_host.submeshes.size(); i++) {
                        int id = m_host.entity_id[i];
			const Submesh *submesh = m_host.submeshes[i];
			const Transform *transform = m_host.submesh_transforms[i];

			const Material &material = Material::all[submesh->material_index];
			if (glm::length(material.emission) > 0
					|| material.has_emission()) {
				_instance_ref ref {transform, submesh, id};
				m_host.emissive_submeshes.insert(ref);
				m_host.emissive_count += submesh->triangles();
			}
		}

		// Update the data
		update_triangle_light_buffers({});
		update_quad_light_buffers(lights, light_transforms);
		update_sbt_data(
			m_host.cachelets,
			m_host.submeshes,
			m_host.submesh_transforms
		);

		// hit_records = &m_host.hit_records;
		m_host.last_updated = clock();

		// Reset the number of samples stored
		m_launch_info.samples = 0;

		/* Update TLAS state
		m_tlas.null = false;
		m_tlas.last_updated = clock(); */

		// Update the status
		updated |= true;
	}

        // If needed, build the TLAS
        if (updated) {
		m_tlas.null = false;
		m_tlas.last_updated = clock();
                m_tlas.handle = m_system->build_tlas(
			renderables,
			m_attachments[m_previous_attachment]->m_hit_group_count
		);
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

        // Update triangle lights that have moved
        std::set <_instance_ref> emissive_to_update;
        for (auto ref : m_host.emissive_submeshes) {
                if (transform_daemon[ref.id])
                        emissive_to_update.insert(ref);
        }

        if (emissive_to_update.size() > 0)
                update_triangle_light_buffers(emissive_to_update);

        // Update host SBT data
        if (transform_daemon.size() > 0) {
                for (int i = 0; i < m_host.submeshes.size(); i++) {
                        int id = m_host.entity_id[i];
                        if (transform_daemon[id]) {
                                m_host.hit_records[i].data.model = m_host.submesh_transforms[i]->matrix();
                                m_host.last_updated = clock();
                        }
                }
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
                handle = m_tlas.handle;
	}

	return {handle, hit_records};
}

// Path tracing computation
void ArmadaRTX::render
                (const ECS &ecs,
                const daemons::Transform &transform_daemon,
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

	auto out = preprocess_scene(ecs, transform_daemon, camera, transform);

	// Reset the accumulation state if needed
	if (!accumulate || out.handle.has_value())
		m_launch_info.samples = 0;

	// Invoke render for current attachment
	auto &attachment = m_attachments[m_previous_attachment];
	attachment->render(this, m_launch_info, out.handle, out.hit_records, m_extent);

	// Increment number of samples
	m_launch_info.samples++;
}

}

}
