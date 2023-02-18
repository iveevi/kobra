// Engine headers
#include "../../include/amadeus/armada.cuh"

#include <nvrtc.h>
#include <filesystem>

// Launch parameters
struct NaivePathTracerParameters : kobra::amadeus::ArmadaLaunchInfo {
	OptixTraversableHandle traversable;

	float *halton_x;
	float *halton_y;

	bool russian_roulette;
};

#ifndef NAIVE_SHADER

using namespace kobra;
using namespace kobra::amadeus;

// Taken from Nvidia's book
static void generate_halton_sequence(int N, int b, std::vector <float> &dst)
{
	int n = 0;
	int d = 1;

	for (int i = 0; i < N; i++) {
		int x = d - n;
		if (x == 1) {
			n = 1;
			d *= b;
		} else {
			int y = d/b;
			while (x <= y)
				y /= b;
			n = (b + 1) * y - x;
		}

		dst[i] = (float) n / (float) d;
	}
}

static void generate_pixel_offsets(int N, std::vector <float> &x, std::vector <float> &y)
{
	static std::vector <int> bases {2, 3, 5, 7, 11, 13};

	srand(time(0));
	int r1 = rand() % bases.size();
	int r2 = rand() % bases.size();

	if (r1 == r2)
		r2 = (r2 + 1) % bases.size();

	int b1 = bases[r1];
	int b2 = bases[r2];

	generate_halton_sequence(N, b1, x);
	generate_halton_sequence(N, b2, y);

	for (int i = 0; i < N; i++) {
		x[i] -= 0.5f;
		y[i] -= 0.5f;
	}
}

// Classic Monte Carlo path tracer
class NaivePathTracer : public AttachmentRTX {
	// SBT record types
	using RaygenRecord = optix::Record <int>;
	using MissRecord = optix::Record <int>;

	// Extent
	vk::Extent2D m_extent;

	// Pipeline related information
	OptixModule m_module;

	OptixProgramGroup m_ray_generation;
	OptixProgramGroup m_closest_hit;
	OptixProgramGroup m_closest_hit_shadow;
	OptixProgramGroup m_miss;
	OptixProgramGroup m_miss_shadow;

	OptixPipeline m_pipeline;

	// Buffer for launch parameters
	NaivePathTracerParameters m_parameters;

	std::vector <float> m_pixel_offsets_x;
	std::vector <float> m_pixel_offsets_y;

	CUdeviceptr m_cuda_parameters;

	// TODO: stream

	// Create the program groups and pipeline
	void create_pipeline(const OptixDeviceContext &optix_context) {
		static constexpr const char OPTIX_PTX_FILE[] =
			"bin/ptx/naive_path_tracer.ptx";

		// Load module
		m_module = optix::load_optix_module(
			optix_context, OPTIX_PTX_FILE,
			ppl_compile_options, module_options
		);

		// Load programs
		OptixProgramGroupOptions program_options = {};

		// Descriptions of all the programs
		std::vector <OptixProgramGroupDesc> program_descs = {
			OPTIX_DESC_RAYGEN(m_module, "__raygen__"),
			OPTIX_DESC_HIT(m_module, "__closesthit__"),
			OPTIX_DESC_MISS(m_module, "__miss__"),
		};

		// Corresponding program groups
		std::vector <OptixProgramGroup *> program_groups = {
			&m_ray_generation,
			&m_closest_hit,
			&m_miss,
		};

		optix::load_program_groups(
			optix_context,
			program_descs,
			program_options,
			program_groups
		);

		m_pipeline = optix::link_optix_pipeline(
			optix_context,
			{
				m_ray_generation,
				m_closest_hit,
				m_miss,
			},
			ppl_compile_options,
			ppl_link_options
		);
	}

	OptixShaderBindingTable m_sbt;
public:
	// Constructor
	NaivePathTracer() : AttachmentRTX(1) {
		m_parameters.russian_roulette = false;
	}

	// Attaching and unloading
	// TODO: return bool to indicate success
	void attach(const ArmadaRTX &armada_rtx) override {
		// First load the pipeline
		create_pipeline(armada_rtx.system()->context());

		// Get the extent
		m_extent = armada_rtx.extent();

		// Allocate the SBT
		std::vector <RaygenRecord> ray_generation_records(1);
		std::vector <MissRecord> miss_records(1);

		// Fill the SBT
		optix::pack_header(m_ray_generation, ray_generation_records[0]);
		optix::pack_header(m_miss, miss_records[0]);

		// Create the SBT
		m_sbt = {};

		m_sbt.raygenRecord = cuda::make_buffer_ptr(ray_generation_records);

		m_sbt.missRecordBase = cuda::make_buffer_ptr(miss_records);
		m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
		m_sbt.missRecordCount = miss_records.size();

		m_sbt.hitgroupRecordBase = 0;
		m_sbt.hitgroupRecordStrideInBytes = 0;
		m_sbt.hitgroupRecordCount = 0;

		// Initialize the parameters buffer
		m_cuda_parameters = (CUdeviceptr) cuda::alloc <NaivePathTracerParameters> (1);

		m_pixel_offsets_x.clear();
		m_pixel_offsets_y.clear();
	}

	void load() override {
		// Generate the pixel offsets
		if (m_pixel_offsets_x.empty() || m_pixel_offsets_y.empty()) {
			int width = m_extent.width;
			int height = m_extent.height;

			m_pixel_offsets_x.resize(width * height);
			m_pixel_offsets_y.resize(width * height);

			generate_pixel_offsets(
				width * height,
				m_pixel_offsets_x, m_pixel_offsets_y
			);

			// Copy the parameters to the GPU
			std::cout << "Copying parameters to GPU\n";
			std::cout << "\tResolution: " << width << 'x' << height << '\n';
			std::cout << "\tsize: " << m_pixel_offsets_x.size() << '\n';
			m_parameters.halton_x = cuda::make_buffer(m_pixel_offsets_x);
			m_parameters.halton_y = cuda::make_buffer(m_pixel_offsets_y);
		}
	}

	void unload() override {
	}

	// Options
	void set_option(const std::string &field, const OptionValue &value) override {
		if (field == "russian_roulette") {
			if (std::holds_alternative <bool> (value)) {
				m_parameters.russian_roulette = std::get <bool> (value);
				std::cout << "Russian roulette: " << std::get <bool> (value) << '\n';
			} else {
				KOBRA_LOG_FILE(Log::WARN) << "Invalid value for"
					" russian_roulette option, expected bool\n";
			}
		}
	}

	OptionValue get_option(const std::string &field) const override {
		if (field == "russian_roulette")
			return m_parameters.russian_roulette;

		return {};
	}

	// Rendering
	void render(const ArmadaRTX *armada_rtx,
			const ArmadaLaunchInfo &launch_info,
			const std::optional <OptixTraversableHandle> &handle,
			std::vector <HitRecord> *hit_records,
			const vk::Extent2D &extent) override {
		// Check if hit groups need to be updated, and update them if necessary
		if (hit_records) {
			// Free old buffer
			if (m_sbt.hitgroupRecordBase)
				cuda::free(m_sbt.hitgroupRecordBase);

			// Update the SBT
			std::vector <HitRecord> local_hit_records(hit_records->size());
			for (size_t i = 0; i < hit_records->size(); i++) {
				local_hit_records[i] = (*hit_records)[i];
				pack_header(m_closest_hit, local_hit_records[i]);
			}

			m_sbt.hitgroupRecordBase = cuda::make_buffer_ptr(local_hit_records);
			m_sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
			m_sbt.hitgroupRecordCount = local_hit_records.size();
		}

		// Copy the parameters and launch
		std::memcpy(&m_parameters, &launch_info, sizeof(ArmadaLaunchInfo));

		if (handle)
			m_parameters.traversable = *handle;

		cuda::copy(m_cuda_parameters, &m_parameters, 1, cudaMemcpyHostToDevice);

		// Execute the pipeline
		OPTIX_CHECK(
			optixLaunch(
				m_pipeline, 0,
				m_cuda_parameters,
				sizeof(NaivePathTracerParameters),
				&m_sbt, extent.width, extent.height, 1
			)
		);

		CUDA_SYNC_CHECK();
	}
};

extern "C" {

struct ret {
	const char *name;
	kobra::amadeus::AttachmentRTX *ptr;
};

ret load_attachment()
{
	std::vector <std::string> include_paths = {
		KOBRA_DIR,
		KOBRA_DIR "/thirdparty/glm",
		KOBRA_DIR "/thirdparty/optix",
		KOBRA_DIR "/thirdparty/termcolor/include",
	};

	// TODO: compile the necessary PTX files
	constexpr char CUDA_SHADER_FILE[] = KOBRA_DIR "/source/optix/naive_path_tracer.cu";

	// Write the PTX file
	const std::string ptx_file = "bin/ptx/naive_path_tracer.ptx";

	std::vector <const char *> options {
		"-std=c++17",
		"-arch=compute_75",
		"-lineinfo",
		"-g",
		"-DKOBRA_OPTIX_SHADER=0",
		"--expt-relaxed-constexpr",
	};

	for (auto &path : include_paths) {
		path = "-I" + path;
		options.push_back(path.c_str());
	}

	std::string flags;
	for (const auto &option : options) {
		std::cout << "Option: " << option << std::endl;
		flags += option;
		flags += " ";
	}

	// Compile the PTX file
	std::string cmd = "nvcc -ptx " + flags;
	cmd += CUDA_SHADER_FILE;
	cmd += " -o " + ptx_file;

	std::cout << "Compiling PTX file: " << cmd << std::endl;
	int ret = system(cmd.c_str());
	if (ret != 0) {
		KOBRA_LOG_FUNC(Log::ERROR) << "Failed to compile PTX file\n";
		return {nullptr, nullptr};
	}

	std::cout << "PTX file compiled successfully\n";

	// TODO: global wide cache for the PTX files...
	return {
		"Naive Path Tracer",
		new NaivePathTracer()
	};
}

// TODO: dellocate attachment

}

#endif
