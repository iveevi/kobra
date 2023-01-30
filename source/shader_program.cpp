#include "../include/shader_program.hpp"

namespace kobra {

// Local structs
struct _compile_out {
	std::vector <unsigned int> 	spirv = {};
	std::string			log = "";
};

// Compiling shaders
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

	KOBRA_LOG_FUNC(Log::ERROR) << "Unknown shader stage: "
		<< vk::to_string(stage) << std::endl;

	return EShLangVertex;
}

_compile_out glsl_to_spriv(const std::string &source, const vk::ShaderStageFlagBits &shader_type)
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

// Constructor
ShaderProgram::ShaderProgram
		(const std::string &source,
		const vk::ShaderStageFlagBits &shader_type)
		: m_source(source), m_shader_type(shader_type) {}

// Compile shader
std::optional <vk::raii::ShaderModule> ShaderProgram::compile
		(const vk::raii::Device &device)
{
	// If has failed before, don't try again
	if (m_failed)
		return std::nullopt;

	// Check that file exists
	glslang::InitializeProcess();

	// Compile shader
	_compile_out out = glsl_to_spriv(m_source, m_shader_type);
	if (!out.log.empty()) {
		KOBRA_LOG_FUNC(Log::ERROR) << "Shader compilation failed:\n"
			<< out.log << std::endl;

		m_failed = true;
		return std::nullopt;
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

}
