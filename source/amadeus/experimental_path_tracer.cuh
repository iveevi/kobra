#pragma once

// Engine headers
#include "../../include/amadeus/armada.cuh"

#include <nvrtc.h>
#include <filesystem>

// Launch parameters
struct ExperimentalPathTracerParameters : kobra::amadeus::ArmadaLaunchInfo {
	OptixTraversableHandle traversable;

	bool russian_roulette;
};
