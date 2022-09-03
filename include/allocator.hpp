#ifndef KOBRA_ALLOCATOR_H_
#define KOBRA_ALLOCATOR_H_

// Standard headers
#include <map>
#include <cstdlib>

// Engine headers
#include "logger.hpp"

namespace kobra {

struct Allocator {
	size_t allocated = 0;
	size_t deallocated = 0;

	// Destuctor does simple leak check
	~Allocator() {
		throw false;
		std::cout << "Allocated: " << allocated << std::endl;
		std::cout << "Deallocated: " << deallocated << std::endl;
		if (allocated != deallocated) {
			KOBRA_LOG_FUNC(Log::WARN) << "Memory leak detected! "
				<< (allocated - deallocated) << " bytes leaked.";
		}
	}

	// TODO: preallocation strategy
	void *alloc(size_t size) {
		allocated += size;
		return malloc(size);
	}

	template <class T>
	T *alloc(size_t count = 1) {
		allocated += sizeof(T) * count;
		return (T *) malloc(sizeof(T) * count);
	}

	// TODO: mark variable as deallocated
	void dealloc(void *ptr, size_t size) {
		deallocated += size;
		free(ptr);
	}

	// Singleton
	static Allocator &one() {
		static Allocator instance;
		return instance;
	}
};

}

#endif
