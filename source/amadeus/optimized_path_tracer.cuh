#pragma once

// Engine headers
#include "../../include/amadeus/armada.cuh"

#include <nvrtc.h>
#include <filesystem>

// Launch parameters
struct OptimizedPathTracerParameters : kobra::amadeus::ArmadaLaunchInfo {
	OptixTraversableHandle traversable;

	bool russian_roulette;
};
