#ifndef SHADER_PROGRAM_H_
#define SHADER_PROGRAM_H_

// Standard headers
#include <thread>
#include <filesystem>

// Glslang and SPIRV-Tools
#include <glslang/SPIRV/GlslangToSpv.h>
#include <glslang/StandAlone/ResourceLimits.h>

// Engine headers
#include "backend.hpp"
#include "common.hpp"

namespace kobra {

// Custom shader programs
class ShaderProgram {
private:
	bool m_failed = false;
	std::string m_source;
	vk::ShaderStageFlagBits	m_shader_type;
public:
	// Default constructor
	ShaderProgram() = default;

	// Constructor
	ShaderProgram(const std::string &, const vk::ShaderStageFlagBits &);

	// Compile shader
	std::optional <vk::raii::ShaderModule> compile(const vk::raii::Device &);
};

}

#endif
