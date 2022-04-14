#ifndef KOBRA_RT_BATCH_H_
#define KOBRA_RT_BATCH_H_

// Standard headers
#include <vector>

// Engine headers
#include "../common.hpp"

namespace kobra {

namespace rt {

// Forward declarations
class Batch;

// Batch index
class BatchIndex {
	Batch *batch = nullptr;
public:
	uint width;
	uint height;

	uint offset_x;
	uint offset_y;

	uint pixel_samples;
	uint light_samples;

	bool accumulate = false;

	// Default constructor
	BatchIndex() = default;

	// Constructor
	BatchIndex(int, int, int, int, int, int);

	// Methods
	void set_batch(Batch *);
	void callback() const;
};

// Batch class
class Batch {
	int width;
	int height;

	int batch_width;
	int batch_height;

	int batches_x;
	int batches_y;

	int batch_x;
	int batch_y;

	int max_samples;

	// int **sample_count = nullptr;
	std::vector <std::vector <int>> sample_count;
public:
	// Default constuctor
	Batch() = default;

	// Constructor
	Batch(int, int, int, int, int);

	// Methods
	BatchIndex make_batch_index(int x, int y, int p = 1, int l = 1);
	
	void increment(BatchIndex &index);
	void increment_sample_count(const BatchIndex &);

	int samples(const BatchIndex &) const;
	int total_samples() const;
	
	bool completed() const;
	void reset();
	float progress() const;
};

}

}

#endif
