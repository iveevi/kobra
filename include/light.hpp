#ifndef LIGHT_H_
#define LIGHT_H_

// Engine headers
#include "core.hpp"
#include "transform.hpp"
#include "types.hpp"
#include "world_update.hpp"

// Light structure
struct Light {
	float		id = LIGHT_TYPE_POINT;
	Transform	transform;
	float		intensity;

	// Light constructors
	Light() {}
	Light(float x, const Transform &t, float i)
			: id(x), transform(t), intensity(i) {}

	// Virtual light destructor
	virtual ~Light() {}

	// Write data to aligned_vec4 buffer
	virtual void write(kobra::WorldUpdate &) const = 0;

	// Write full light data
	void write_light(kobra::WorldUpdate &wu) {
		// Push ID, then everythig else
		glm::vec3 pos = transform.position();

		wu.bf_lights->push_back(aligned_vec4 {
			glm::vec4 {
				id, pos.x,
				pos.y,
				pos.z
			}
		});

		this->write(wu);
	}
};

// Point light
struct PointLight : public Light {
	PointLight() {}
	PointLight(const Transform &t, float i)
			: Light(LIGHT_TYPE_POINT, t, i) {}

	void write(kobra::WorldUpdate &) const override {}
};

#endif
