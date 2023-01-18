#include "../include/allocator.hpp"

// Overloaded new and delete operators
void *operator new(size_t size) {
	return kobra::Allocator::one().alloc(size);
}

void *operator new[](size_t size) {
	return kobra::Allocator::one().alloc(size);
}

void operator delete(void *ptr, size_t size) {
	kobra::Allocator::one().dealloc(ptr, size);
}

void operator delete[](void *ptr, size_t size) {
	kobra::Allocator::one().dealloc(ptr, size);
}

