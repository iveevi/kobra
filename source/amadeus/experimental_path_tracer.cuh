#pragma once

// Engine headers
#include "../../include/amadeus/armada.cuh"

#include <nvrtc.h>
#include <filesystem>

// Launch parameters
struct ExperimentalPathTracerParameters : kobra::amadeus::ArmadaLaunchInfo {
	OptixTraversableHandle traversable;

	float *halton_x;
	float *halton_y;

	bool russian_roulette;

	int instances;
};
