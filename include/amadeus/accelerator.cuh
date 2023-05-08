#pragma once

// Standard headers
#include <map>
#include <vector>

// OptiX headers
#include <optix.h>

// Engine headers
#include "include/cuda/alloc.cuh"
#include "include/cuda/cast.cuh"
#include "include/daemons/transform.hpp"
#include "include/system.hpp"
#include "include/optix/core.cuh"
#include "include/renderable.hpp"

namespace kobra {

namespace amadeus {

// Backend management for raytacing
class Accelerator {
        // Reference to System transform daemon
        // TODO: embed in the System itself
        kobra::daemons::Transform *transform_daemon = nullptr;

        // Critical OptiX objects
        OptixDeviceContext m_context = 0;

        // Instance type
        struct Instance {
                // const Renderable *m_renderable = nullptr;
                int index = 0;
                glm::mat4 transform;
                OptixTraversableHandle m_gas = 0;
        };

        // Object cache
        // TODO: callback system to upate cache
        struct {
                std::map <int, std::vector <Instance>> instances;
        } m_cache;

        // Build GAS for an instance
        void build_gas(const Entity &entity, Instance &instance) {
                // Build acceleration structures
                OptixAccelBuildOptions gas_accel_options = {};
                gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
                gas_accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

                // Flags
                const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

                // TODO: import buffers later
                auto &renderable = entity.get <Renderable> ();
		const Submesh &s = renderable.mesh->submeshes[instance.index];

		// Prepare submesh vertices and triangles
		std::vector <float3> vertices;
		std::vector <uint3> triangles;

		// TODO: cache CUDA buffers from either vertex buffer (with stride)
		for (int j = 0; j < s.indices.size(); j += 3) {
			triangles.push_back({
				s.indices[j],
				s.indices[j + 1],
				s.indices[j + 2]
			});
		}

		for (int j = 0; j < s.vertices.size(); j++) {
			auto p = s.vertices[j].position;
			vertices.push_back(cuda::to_f3(p));
		}

		// Create the build input
		OptixBuildInput build_input {};

		build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		CUdeviceptr d_vertices = cuda::make_buffer_ptr(vertices);
		CUdeviceptr d_triangles = cuda::make_buffer_ptr(triangles);

		OptixBuildInputTriangleArray &triangle_array = build_input.triangleArray;
		triangle_array.vertexFormat	= OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_array.numVertices	= vertices.size();
		triangle_array.vertexBuffers	= &d_vertices;

		triangle_array.indexFormat	= OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangle_array.numIndexTriplets	= triangles.size();
		triangle_array.indexBuffer	= d_triangles;

		triangle_array.flags		= triangle_input_flags;

		// SBT record properties
		triangle_array.numSbtRecords	= 1;
		triangle_array.sbtIndexOffsetBuffer = 0;
		triangle_array.sbtIndexOffsetStrideInBytes = 0;
		triangle_array.sbtIndexOffsetSizeInBytes = 0;

		// Build GAS
		CUdeviceptr d_gas_output;
		CUdeviceptr d_gas_tmp;

		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(
			optixAccelComputeMemoryUsage(
				m_context, &gas_accel_options,
				&build_input, 1,
				&gas_buffer_sizes
			)
		);

		d_gas_output = cuda::alloc(gas_buffer_sizes.outputSizeInBytes);
		d_gas_tmp = cuda::alloc(gas_buffer_sizes.tempSizeInBytes);

		OPTIX_CHECK(
			optixAccelBuild(m_context,
				0, &gas_accel_options,
				&build_input, 1,
				d_gas_tmp, gas_buffer_sizes.tempSizeInBytes,
				d_gas_output, gas_buffer_sizes.outputSizeInBytes,
				&instance.m_gas, nullptr, 0
			)
		);

		// Free data at the end
		cuda::free(d_vertices);
		cuda::free(d_triangles);
		cuda::free(d_gas_tmp);
        }
public:
        // Default constructor
        Accelerator(kobra::daemons::Transform *td)
                : transform_daemon(td), m_context(optix::make_context()) {}

	// Propreties
	OptixDeviceContext context() const {
		return m_context;
	}

        // Update from list of valid entities
        // NOTE: All entities must have a Renderable component
        bool update(const std::vector <Entity> &entities) {
		bool updated = false;

                for (const Entity &entity : entities) {
                        int id = entity.id;
                        auto &renderable = entity.get <Renderable> ();
                        auto &transform = entity.get <Transform> ();

                        // If already cached, just update transform
                        if (m_cache.instances.count(id) > 0) {
                                for (Instance &instance : m_cache.instances[id]) {
                                        if (transform_daemon->changed(id)) {
                                                instance.transform = transform.matrix();
                                                updated |= true;
                                        }
                                }

                                continue;
                        }

                        std::cout << "New renderable: " << &renderable << std::endl;
                        
                        // TODO: if any of the transforms have changed,
                        // then signal an update...

                        // Lazily generate GAS by callback
                        // TODO: callback system
                        int submeshes = renderable.size();

			std::cout << "\t# of submeshes: " << submeshes << std::endl;

                        std::vector <Instance> instances;
                        instances.reserve(submeshes);

                        for (int j = 0; j < submeshes; j++) {
                                // TODO: build GAS for now...
                                instances.emplace_back(Instance {j, transform.matrix()});
                                build_gas(entity, instances.back());
                        }

			m_cache.instances.insert({id, instances});
			updated = true;
                }

		return updated;
        }

        // Build TLAS from selected renderables
        // TODO: choose a subset of entities to render
        // TODO: pass a map of {id, submesh index} -> sbt offset
        using MeshIndex = std::pair <int, int>;

        OptixTraversableHandle build_tlas(int hit_groups, const std::map <MeshIndex, int> &offsets = {}, int mask = 0xFF) {
                std::vector <OptixInstance> optix_instances;

                for (auto pr : m_cache.instances) {
                        for (const Instance &instance : pr.second) {
                                glm::mat4 mat = instance.transform;

                                float transform[12] = {
                                        mat[0][0], mat[1][0], mat[2][0], mat[3][0],
                                        mat[0][1], mat[1][1], mat[2][1], mat[3][1],
                                        mat[0][2], mat[1][2], mat[2][2], mat[3][2]
                                };

                                OptixInstance optix_instance {};
                                memcpy(optix_instance.transform, transform, sizeof(float) * 12);

                                // Set the instance handle
                                optix_instance.traversableHandle = instance.m_gas;
                                optix_instance.visibilityMask = mask;

                                int id = optix_instances.size();

                                // If the map is available, use it only
                                MeshIndex index {pr.first, instance.index};
                                if (offsets.size() > 0)
                                        id = offsets.at(index);

                                optix_instance.instanceId = optix_instances.size();
                                optix_instance.sbtOffset = optix_instances.size() * hit_groups;

                                optix_instances.push_back(optix_instance);
                        }
                }

                // Create top level acceleration structure
                CUdeviceptr d_instances = cuda::make_buffer_ptr(optix_instances);

                OptixBuildInput ias_build_input {};
                ias_build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
                ias_build_input.instanceArray.instances = d_instances;
                ias_build_input.instanceArray.numInstances = optix_instances.size();

                // IAS options
                OptixAccelBuildOptions ias_accel_options {};
                ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
                ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

                // IAS buffer sizes
                OptixAccelBufferSizes ias_buffer_sizes;
                OPTIX_CHECK(
                        optixAccelComputeMemoryUsage(
                                m_context, &ias_accel_options,
                                &ias_build_input, 1,
                                &ias_buffer_sizes
                        )
                );

                // Allocate the IAS
                CUdeviceptr d_ias_output = cuda::alloc(ias_buffer_sizes.outputSizeInBytes);
                CUdeviceptr d_ias_tmp = cuda::alloc(ias_buffer_sizes.tempSizeInBytes);

                // Build the IAS
                OptixTraversableHandle handle;

                OPTIX_CHECK(
                        optixAccelBuild(m_context,
                                0, &ias_accel_options,
                                &ias_build_input, 1,
                                d_ias_tmp, ias_buffer_sizes.tempSizeInBytes,
                                d_ias_output, ias_buffer_sizes.outputSizeInBytes,
                                &handle, nullptr, 0
                        )
                );

                cuda::free(d_ias_tmp);
                cuda::free(d_instances);

		std::cout << "Built TLAS with " << optix_instances.size() << " instances" << std::endl;

                return handle;
        }
};

}

}
