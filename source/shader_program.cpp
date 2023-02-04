// Standard headers
#include <regex>

// Engine headers
#include "../include/shader_program.hpp"

namespace kobra {

// Local structs
struct _compile_out {
	std::vector <unsigned int> 	spirv = {};
	std::string			log = "";
	std::string			source = "";
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

// Includer class
static constexpr const char KOBRA_SHADER_INCLUDE_DIR[] = KOBRA_DIR "/source/shaders/";

std::string preprocess
		(const std::string &source,
		const std::map <std::string, std::string> &defines)
{
	// Defines contains string values to be relpaced
	// e.g. if {"VERSION", "450"} is in defines, then
	// "${VERSION}" will be replaced with "450"

	std::string out = "";
	std::string line;

	std::istringstream stream(source);
	while (std::getline(stream, line)) {
		// Check if line is an include but not commented out
		if (line.find("#include") != std::string::npos &&
				line.find("//") == std::string::npos) {
			// Get the include path
			std::regex regex("#include \"(.*)\"");
			std::smatch match;
			std::regex_search(line, match, regex);

			// Check that the regex matched
			if (match.size() != 2) {
				KOBRA_LOG_FUNC(Log::ERROR)
					<< "Failed to match regex for include: "
					<< line << std::endl;
				continue;
			}

			// Read the file
			std::string path = KOBRA_SHADER_INCLUDE_DIR + match[1].str();
			std::string source = common::read_file(path);

			// Replace the include with the file contents
			out += preprocess(source, defines);

			// TODO: allow simoultaneous features
			// e.g. add the includes file lines into the stream...
		} else if (line.find("${") != std::string::npos) {
			// Replace the define
			for (auto &define : defines) {
				std::string key = "${" + define.first + "}";
				std::string value = define.second;

				// Replace all instances of the key with the value
				size_t pos = 0;
				while ((pos = line.find(key, pos)) != std::string::npos) {
					line.replace(pos, key.length(), value);
					pos += value.length();
				}
			}

			out += line + "\n";
		} else {
			out += line + "\n";
		}
	}

	return out;
}

_compile_out glsl_to_spriv
		(const std::string &source,
		const std::map <std::string, std::string> &defines,
		const vk::ShaderStageFlagBits &shader_type)
{
	std::string source_copy = preprocess(source, defines);

	// Output
	_compile_out out;

	// Compile shader
	EShLanguage stage = translate_shader_stage(shader_type);

	const char *shaderStrings[1];
	shaderStrings[0] = source_copy.data();

	glslang::TShader shader(stage);
	shader.setStrings(shaderStrings, 1);

	// Enable SPIR-V and Vulkan rules when parsing GLSL
	EShMessages messages = (EShMessages) (EShMsgSpvRules | EShMsgVulkanRules);
	// ShaderIncluder includer;
	if (!shader.parse(&glslang::DefaultTBuiltInResource,
			450, false, messages)) {
		out.log = shader.getInfoLog();
		out.source = source_copy;
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
		(const vk::raii::Device &device,
		const std::map <std::string, std::string> &defines)
{
	// If has failed before, don't try again
	if (m_failed)
		return std::nullopt;

	// Check that file exists
	glslang::InitializeProcess();

	// Compile shader
	_compile_out out = glsl_to_spriv(m_source, defines, m_shader_type);
	if (!out.log.empty()) {
		// TODO: show the errornous line(s)
		KOBRA_LOG_FUNC(Log::ERROR)
			<< "Shader compilation failed:\n" << out.log
			<< "\nSource:\n" << out.source << "\n";

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
