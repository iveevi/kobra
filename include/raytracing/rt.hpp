#ifndef KOBRA_RT_H_
#define KOBRA_RT_H_

// Standard headers
#include <memory>

// Engine headers
#include "../backend.hpp"
#include "../buffer_manager.hpp"

namespace kobra {

namespace rt {

// Push constant structure
struct PushConstants {
	alignas(16)
	uint triangles;
	uint lights;
	uint samples_per_pixel;
	uint samples_per_light;

	aligned_vec4 camera_position;
	aligned_vec4 camera_forward;
	aligned_vec4 camera_up;
	aligned_vec4 camera_right;

	aligned_vec4 camera_tunings;
};

// Latching packet
struct LatchingPacket {
	Buffer4f		*vertices;
	Buffer4f		*triangles;
	Buffer4f		*materials;
	Buffer4m		*transforms;
	Buffer4f		*lights;
	BufferManager <uint>	*light_indices;
};

// Element type
struct _element {
	// Destructor
	virtual ~_element() = default;

	// Latch to layer
	virtual void latch(const LatchingPacket &packet, size_t) = 0;
};

// Memory safe
using Element = std::shared_ptr <_element>;

}

}

#endif
