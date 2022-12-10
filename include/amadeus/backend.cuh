#ifndef KOBRA_AMADEUS_BACKEND_H_
#define KOBRA_AMADEUS_BACKEND_H_

// Standard headers
#include <map>

// OptiX headers
#include <optix.h>

// Engine headers
#include "../renderer.hpp"
#include "../optix/core.cuh"
#include "../ecs.hpp"

namespace kobra {

namespace amadeus {

// Backend management for raytacing
class System {
        // Critical OptiX objects
        OptixDeviceContext m_context;

        // Instance type
        struct Instance {
                int m_index = 0;
                glm::mat4 m_transform;
                OptixTraversableHandle m_gas = 0;

                Instance(int index, const Transform *transform)
                                : m_index(index), m_transform(transform->matrix()) {}
        };

        // Object cache
        // TODO: callback system to upate cache
        struct {
                std::map <
                        const Rasterizer *,
                        std::vector <Instance>
                > rasterizers;
        } m_cache;
public:
        // Default constructor
        System() = default;

        // Constructor
        System() : m_context(optix::make_context()) {}

        // Update from ECS
        void update(const ECS &ecs) {
                for (int i = 0; i < ecs.size(); i++) {
                        // TODO: one unifying renderer component, with options for raytracing, etc
                        if (!ecs.exists <Rasterizer> (i))
                                continue;

                        const Rasterizer *rasterizer = &ecs.get <Rasterizer> (i);                                
                        const Transform *transform = &ecs.get <Transform> (i);

                        // If already cached, just update transform
                        if (m_cache.rasterizers.count(rasterizer) > 0) {
                                for (Instance &instance : m_cache.rasterizers[rasterizer])
                                        instance.transform = transform->matrix();

                                continue;
                        }

                        // Lazily generate GAS by callback
                        // TODO: callback system
                        int submeshes = rasterizer->size();
                        
                        std::vector <Instance> instances;
                        instances.reserve(submeshes);

                        for (int j = 0; j < submeshes; j++)
                                instances[i] = Instance(j, transform);

                        m_cache.rasterizers[rasterizer] = instances;
                }
        }
};

}

}

#endif