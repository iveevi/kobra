#ifndef KOBRA_ASMODEUS_WSRIS_H_
#define KOBRA_ASMODEUS_WSRIS_H_

// Engine headers
#include "backend.cuh"

namespace kobra {

namespace asmodeus {

struct WorldSpaceKdResampling {
	// Reference to the backend
	const Backend *backend = nullptr;

	// Construction
	static WorldSpaceKdResampling make(const Backend &backend) {
		WorldSpaceKdResampling wsris;

		wsris.backend = &backend;
		
		return wsris;
	}
};

}

}

#endif
