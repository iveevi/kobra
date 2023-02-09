#ifndef KOBRA_AMADEUS_BACKEND_H_
#define KOBRA_AMADEUS_BACKEND_H_

// Standard headers
#include <map>
#include <vector>

// OptiX headers
#include <optix.h>

// Engine headers
#include "../cuda/alloc.cuh"
#include "../cuda/cast.cuh"
#include "../ecs.hpp"
#include "../optix/core.cuh"
#include "../renderable.hpp"

namespace kobra {

namespace amadeus {

// Backend management for raytacing
class System {
        // Critical OptiX objects
        OptixDeviceContext m_context = 0;

        // Instance type
        struct Instance {
                const Renderable *m_renderable = nullptr;
                int m_index = 0;
                glm::mat4 m_transform;
                OptixTraversableHandle m_gas = 0;

                Instance(const Renderable *source, int index, const Transform *transform)
                                : m_renderable(source), m_index(index),
                                m_transform(transform->matrix()) {}
        };

        // Object cache
        // TODO: callback system to upate cache
        struct {
                std::map <
                        const Renderable *,
                        std::vector <Instance>
                > instances;
        } m_cache;

        // Build GAS for an instance
        void build_gas(Instance &instance) {
                // Build acceleration structures
                OptixAccelBuildOptions gas_accel_options = {};
                gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
                gas_accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
                
                // Flags
                const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        
                // TODO: import buffers later
		const Submesh &s = instance.m_renderable
                        ->mesh->submeshes[instance.m_index];

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
        System() : m_context(optix::make_context()) {}

	// Propreties
	OptixDeviceContext context() const {
		return m_context;
	}

        // Update from ECS
        bool update(const ECS &ecs) {
		bool updated = false;

                for (int i = 0; i < ecs.size(); i++) {
                        // TODO: one unifying renderer component, with options for raytracing, etc
                        if (!ecs.exists <Renderable> (i))
                                continue;

                        const Renderable *renderable = &ecs.get <Renderable> (i);                                
                        const Transform *transform = &ecs.get <Transform> (i);

                        // If already cached, just update transform
                        if (m_cache.instances.count(renderable) > 0) {
                                for (Instance &instance : m_cache.instances[renderable])
                                        instance.m_transform = transform->matrix();

                                continue;
                        }

                        // Lazily generate GAS by callback
                        // TODO: callback system
                        int submeshes = renderable->size();

			std::cout << "\t# of submeshes: " << submeshes << std::endl;
                        
                        std::vector <Instance> instances;
                        instances.reserve(submeshes);

                        for (int j = 0; j < submeshes; j++) {
                                // TODO: build GAS for now...
                                instances.emplace_back(Instance(renderable, j, transform));
                                build_gas(instances.back());
                        }

			m_cache.instances.insert({renderable, instances});
			updated = true;
                }

		return updated;
        }

        // Build TLAS from selected renderables
        OptixTraversableHandle build_tlas(const std::vector <const Renderable *> &renderables, int hit_groups, int mask = 0xFF) {
                std::vector <OptixInstance> optix_instances;

                for (const Renderable *renderable : renderables) {
                        const std::vector <Instance> &instances = m_cache.instances[renderable];

                        for (const Instance &instance : instances) {
                                glm::mat4 mat = instance.m_transform;

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
                                optix_instance.sbtOffset = optix_instances.size() * hit_groups;
                                optix_instance.instanceId = optix_instances.size();

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

                return handle;
        }
};

}

}

#endif
