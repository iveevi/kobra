#ifndef KOBRA_MATERIAL_H_
#define KOBRA_MATERIAL_H_

// Standard headers
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>

// Engine headers
// #include "backend.hpp"
#include "common.hpp"
#include "core.hpp"
#include "types.hpp"

namespace kobra {

// Materials, in GGX form
struct Material {
        // Name of the material
        std::string name;

	// TODO: default should be purple, flat shading
	glm::vec3	diffuse {1, 0, 1};
	glm::vec3	specular {0.0f};
	glm::vec3	emission {0.0f};
	glm::vec3	ambient {0.2f};
	float		shininess {0.0f};
	float		roughness {0.0f};
	float		refraction {1.0f};

	// TODO: extinction, absorption, etc.

	// TODO: refactor ttexture names (diffuse, etc)
	std::string	albedo_texture = "";
	std::string	normal_texture = "";
	std::string	specular_texture = "";
	std::string	emission_texture = "";
	std::string	roughness_texture = "";

	// TODO: emissive termm, reafctor eEmissive to eLight?
	Shading		type = Shading::eDiffuse;

	// TODO: each material needs a reference to a shader program...
	// how do we extend it to support multiple shader languages, like CUDA?
	// Material presets:
	// 	GGX, Disney, Blinn-Phong based materials?
	// 	Then restrict CUDA to GGX only?
	//	Actually cuda can implement all these, and then extract
	//	appropriate during calculate_material()

	// Properties
	// TODO: refactor to better names...
	bool has_albedo() const;
	bool has_normal() const;
	bool has_specular() const;
	bool has_emission() const;
	bool has_roughness() const;

	// Serialize to GPU buffer
	void serialize(std::vector <aligned_vec4> &) const;

	// Global material list
	// TODO: one list per scene
	static std::vector <Material> all;
	
	// Management daemon
	static struct Daemon {
		using Addresses = std::set <uint32_t>;
		using Pinger = std::function <void (void *, const Addresses &)>;

		struct PingData {
			void *user;
			Pinger pinger;
			Addresses indices;
		};

		std::vector <PingData> m_pingers;

		void ping_at(void *user, const Pinger &pinger) {
			m_pingers.push_back({user, pinger, {}});
		}

		void update(uint32_t index) {
			for (auto &pinger : m_pingers)
				pinger.indices.insert(index);
		}

		// Call at the end of each frame
		void ping_all() {
			for (auto &pinger : m_pingers) {
				if (pinger.indices.empty())
					continue;

				pinger.pinger(pinger.user, pinger.indices);
				pinger.indices.clear();
			}
		}
	} daemon;
};

}

#endif
