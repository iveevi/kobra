#ifndef SHADER_PROGRAM_H_
#define SHADER_PROGRAM_H_

// Glslang and SPIRV-Tools
#include <glslang/SPIRV/GlslangToSpv.h>
#include <glslang/StandAlone/ResourceLimits.h>

// Engine headers
#include "backend.hpp"
#include "common.hpp"

namespace kobra {

// Compiling shaders
// TODO: source
inline EShLanguage translate_shader_stage(const vk::ShaderStageFlagBits &stage)
{
	switch (stage) {
	case vk::ShaderStageFlagBits::eVertex:
		return EShLangVertex;
	case vk::ShaderStageFlagBits::eTessellationControl:
		return EShLangTessControl;
	case vk::ShaderStageFlagBits::eTessellationEvaluation:
		return EShLangTessEvaluation;
	case vk::ShaderStageFlagBits::eGeometry:
		return EShLangGeometry;
	case vk::ShaderStageFlagBits::eFragment:
		return EShLangFragment;
	case vk::ShaderStageFlagBits::eCompute:
		return EShLangCompute;
	case vk::ShaderStageFlagBits::eRaygenNV:
		return EShLangRayGenNV;
	case vk::ShaderStageFlagBits::eAnyHitNV:
		return EShLangAnyHitNV;
	case vk::ShaderStageFlagBits::eClosestHitNV:
		return EShLangClosestHitNV;
	case vk::ShaderStageFlagBits::eMissNV:
		return EShLangMissNV;
	case vk::ShaderStageFlagBits::eIntersectionNV:
		return EShLangIntersectNV;
	case vk::ShaderStageFlagBits::eCallableNV:
		return EShLangCallableNV;
	case vk::ShaderStageFlagBits::eTaskNV:
		return EShLangTaskNV;
	case vk::ShaderStageFlagBits::eMeshNV:
		return EShLangMeshNV;
	default:
		break;
	}

	KOBRA_LOG_FUNC(error) << "Unknown shader stage: "
		<< vk::to_string(stage) << std::endl;
	return EShLangVertex;
}

struct _compile_out {
	std::vector <unsigned int> 	spirv = {};
	std::string			log = "";
};

inline _compile_out glsl_to_spriv(const std::string &source, const vk::ShaderStageFlagBits &shader_type)
{
	// Output
	_compile_out out;

	// Compile shader
	EShLanguage stage = translate_shader_stage(shader_type);

	const char * shaderStrings[1];
	shaderStrings[0] = source.data();

	glslang::TShader shader( stage );
	shader.setStrings( shaderStrings, 1 );

	// Enable SPIR-V and Vulkan rules when parsing GLSL
	EShMessages messages = (EShMessages) (EShMsgSpvRules | EShMsgVulkanRules);

	if (!shader.parse( &glslang::DefaultTBuiltInResource, 100, false, messages)) {
		out.log = shader.getInfoLog();
		return out;
	}

	// Link the program
	glslang::TProgram program;
	program.addShader(&shader);

	if (!program.link(messages)) {
		out.log = program.getInfoLog();
		return out;
	}

	glslang::GlslangToSpv(*program.getIntermediate(stage), out.spirv);
	return out;
}

// Forward declarations
namespace layers {

class ShapeRenderer;

}

// Custom shader programs
//	either fragment or compute shader
class ShaderProgram {
	// Pointer to pipeline
	vk::raii::Pipeline	*_pipeline = nullptr;
	bool			_cc_failed = false;

	// Shader source or file
	std::string		_source = "";
	std::string		_file = "";
public:
	// TODO: shader class for including different inputs
	// 	(i.e. shapes, text, meshes, comute,etc.)

	// Default constructor
	ShaderProgram() = default;

	// Copy does everything except pipeline and cc_failed
	ShaderProgram(const ShaderProgram &other)
		: _source(other._source), _file(other._file) {}

	// Set shader source
	void set_source(const std::string &source) {
		_source = source;
	}

	// Set shader file
	void set_file(const std::string &file) {
		_file = file;
	}

	// Check if it is a valid shader
	bool valid() const {
		return !(_source.empty() && _file.empty());
	}

	// Check if it has failed to compile before
	bool failed() const {
		return _cc_failed;
	}

	// Compile shader
	// TODO: deal with imports, etc here
	std::optional <vk::raii::ShaderModule> compile(const vk::raii::Device &device) {
		glslang::InitializeProcess();

		if (!valid()) {
			KOBRA_LOG_FUNC(warn) << "Shader program is invalid, returning nullptr\n";
			return {};
		}

		std::string source = _source;
		if (_source.empty())
			source = common::read_file(_file);

		// Compile shader
		_compile_out out = glsl_to_spriv(source, vk::ShaderStageFlagBits::eFragment);
		if (!out.log.empty()) {
			KOBRA_LOG_FUNC(error) << "Shader compilation failed:\n"
				<< out.log << std::endl;

			_cc_failed = true;
			return {};
		}

		// Create shader module
		return vk::raii::ShaderModule(
			device,
			vk::ShaderModuleCreateInfo(
				vk::ShaderModuleCreateFlags(),
				out.spirv
			)
		);
	}

	friend class layers::ShapeRenderer;
};

}

#endif
