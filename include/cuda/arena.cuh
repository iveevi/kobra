#ifndef KOBRA_CUDA_ARENA_H_
#define KOBRA_CUDA_ARENA_H_

namespace kobra {

namespace cuda {

// Memory arena for CUDA device memory, so that we can allocate and free
// memory without the overhead of cudaMalloc and cudaFree.
class Arena {
public:
};

}

}

#endif
