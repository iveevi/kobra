#ifndef WORLD_UPDATE_H_
#define WORLD_UPDATE_H_

// Engine headers
#include "core.hpp"

namespace mercury {

// Update structure
struct WorldUpdate {
	Buffer		objects;
	Buffer		materials;
	Buffer		lights;
	Indices		indices;
};

}

#endif