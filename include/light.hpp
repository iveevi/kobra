#ifndef LIGHT_H_
#define LIGHT_H_

// Engine headers
#include "core.hpp"
#include "types.hpp"

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
	virtual void write(Buffer &buffer) const = 0;

	// Write full light data
	void write_to_buffer(Buffer &buffer) {
		// Push ID, then everythig else
		buffer.push_back(aligned_vec4 {
			glm::vec4 {
				id, transform.position.x,
				transform.position.y,
				transform.position.z
			}
		});

		this->write(buffer);
	}
};

// Point light
struct PointLight : public Light {
	PointLight() {}
	PointLight(const Transform &t, float i)
			: Light(LIGHT_TYPE_POINT, t, i) {}

	void write(Buffer &buffer) const override {}
};

#endif