#ifndef WORLD_UPDATE_H_
#define WORLD_UPDATE_H_

// Engine headers
#include "core.hpp"
// #include "buffer_manager.hpp"

namespace kobra {

// Update structure
struct WorldUpdate {
	Buffer4f *	bf_objs;
	Buffer4f *	bf_lights;
	Buffer4f *	bf_mats;
	Buffer4f *	bf_verts;
	Buffer4m *	bf_trans;
	
	Indices		indices;
};

}

#endif
