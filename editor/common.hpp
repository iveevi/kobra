#pragma once

// Engine headers
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/layers/common.hpp"
#include "include/layers/forward_renderer.hpp"
#include "include/layers/image_renderer.hpp"
#include "include/layers/objectifier.hpp"
#include "include/layers/ui.hpp"
#include "include/project.hpp"
#include "include/scene.hpp"
#include "include/shader_program.hpp"
#include "include/ui/attachment.hpp"
#include "include/engine/irradiance_computer.hpp"
#include "include/amadeus/armada.cuh"
#include "include/amadeus/path_tracer.cuh"
#include "include/amadeus/restir.cuh"
#include "include/layers/framer.hpp"
#include "include/cuda/color.cuh"
#include "include/layers/denoiser.cuh"
#include "include/daemons/transform.hpp"

// Native File Dialog
#include <nfd.h>

// ImPlot headers
#include <implot/implot.h>
#include <implot/implot_internal.h>

// ImGuizmo
#include <ImGuizmo/ImGuizmo.h>

// Extra GLM headers
#include <glm/gtc/type_ptr.hpp>

// Aliasing declarations
using namespace kobra;

// Global communications structure
struct Application {
	Context context;
	float speed = 10.0f;
} g_application;

// Rasterizing for primary intersection and G-buffer
struct PrimaryRasterizer {

};

// Editor rendering
struct EditorRenderer {

};