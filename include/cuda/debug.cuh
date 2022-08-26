#ifndef KOBRA_CUDA_DEBUG_H_
#define KOBRA_CUDA_DEBUG_H_

#undef printf

#ifdef KCUDA_DEBUG

#define print(...) printf(__VA_ARGS__)

#else

#define assert(...)
#define print(...)

#endif

#endif
