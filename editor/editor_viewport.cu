// Vulkan headers
#include <vulkan/vulkan_format_traits.hpp>
#include <vulkan/vulkan_structs.hpp>

// Engine headers
#include "include/cuda/error.cuh"
#include "include/cuda/interop.cuh"
#include "include/daemons/mesh.hpp"

// Editor headers
#include "editor_viewport.cuh"

// Environment map
void load_environment_map(EnvironmentMap *em, kobra::TextureLoader *loader, const std::filesystem::path &path)
{
        const ImageData &img = loader->load_texture(path);
        em->texture = kobra::cuda::import_vulkan_texture(*loader->m_device.device, img);
        em->valid = true;
}

// Constructor
EditorViewport::EditorViewport(const Context &context,
                const std::shared_ptr <amadeus::Accelerator> &_system,
                const std::shared_ptr <MeshDaemon> &_mesh_memory,
                MaterialDaemon *md)
                : system(_system),
                mesh_memory(_mesh_memory),
                phdev(context.phdev),
                device(context.device),
                descriptor_pool(context.descriptor_pool),
                command_pool(context.command_pool),
                texture_loader(context.texture_loader)
{
        common_rtx.timer.start();

        path_tracer.launch_params.color = 0;
        path_tracer.depth = 2;
       
        // TODO: move to default constructor
        // sparse_gi.launch_params.color = 0;
        sparse_gi.launch_params.previous_position = nullptr;
        sparse_gi.depth = 2;

        // amadeus_path_tracer.depth = 2;
        mamba = new Mamba(system->context());
        resize(context.extent);

        configure_present();

        configure_gbuffer_pipeline();
        configure_albedo_pipeline();
        configure_normals_pipeline();
        configure_uv_pipeline();
        configure_triangulation_pipeline();
        configure_bounding_box_pipeline();
        configure_sobel_pipeline();
        configure_highlight_pipeline();

        configure_gbuffer_rtx();
        initialize(&sparse_gi, system->context());
        // configure_amadeus_path_tracer(context);
        configure_path_tracer(context);
      
        // Common raytracing data
        common_rtx.framer = kobra::layers::Framer(context, present_render_pass);

        render_state.initialized = true;

        // Daemon interactions
        subscribe(md, &common_rtx.material_update_queue);
}

// Destructor
EditorViewport::~EditorViewport()
{
        if (framebuffer_images)
                delete framebuffer_images;
        if (mamba)
                delete mamba;
}

static void import_vulkan_texture
                (const vk::raii::Device &device,
                const ImageData &image,
                cudaTextureObject_t &texture,
                cudaSurfaceObject_t &surface,
                cudaChannelFormatDesc &channel_desc)
{
        vk::Extent2D extent = image.extent;

        cudaExternalMemoryHandleDesc cuda_handle_desc {};
        cuda_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        cuda_handle_desc.handle.fd = image.get_memory_handle(device);
	cuda_handle_desc.size = image.get_size();

	// Import the external memory
	cudaExternalMemory_t tex_mem {};
	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaImportExternalMemory(&tex_mem, &cuda_handle_desc));

        // Load the mip maps
        cudaExternalMemoryMipmappedArrayDesc cuda_mipmapped_array_desc {};
        cuda_mipmapped_array_desc.offset = 0;
        cuda_mipmapped_array_desc.formatDesc = channel_desc;
        cuda_mipmapped_array_desc.numLevels = 1;
        cuda_mipmapped_array_desc.flags = 0;
        cuda_mipmapped_array_desc.extent = make_cudaExtent(extent.width, extent.height, 0);

        // Create a CUDA mipmapped array
        cudaMipmappedArray_t mipmap_array {};
        CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(
                &mipmap_array,
                tex_mem, &cuda_mipmapped_array_desc
        ));

        // Get first level of the mipmap
        cudaArray_t cuda_array {};
        CUDA_CHECK(cudaGetMipmappedArrayLevel(
                &cuda_array, mipmap_array, 0
        ));

        // Creat the surface object
        cudaResourceDesc res_desc {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;

        CUDA_CHECK(cudaCreateSurfaceObject(&surface, &res_desc));

        // Also a CUDA texture object for sampling
	res_desc.resType = cudaResourceTypeMipmappedArray;
	res_desc.res.mipmap.mipmap = mipmap_array;

	cudaTextureDesc tex_desc {};
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = true;
	tex_desc.filterMode = cudaFilterModeLinear;

        // Discrete sampling if the texture is not floating point
        if (channel_desc.f != cudaChannelFormatKindFloat)
                tex_desc.filterMode = cudaFilterModePoint;
        
        CUDA_CHECK(cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr));
}

void EditorViewport::resize(const vk::Extent2D &new_extent)
{
        // Update extent
        extent = new_extent;

        KOBRA_LOG_FUNC(Log::INFO) << "Resizing viewport to " << new_extent.width << "x" << new_extent.height << std::endl;
        if (framebuffer_images != nullptr)
                delete framebuffer_images;

        framebuffer_images = new FramebufferResources(*phdev, *device, new_extent);

        // TODO: cuda free the surfaces...

        // Allocate resources for raytracing pipelines
        if (common_rtx.dev_color != 0)
                CUDA_CHECK(cudaFree(common_rtx.dev_color));

        CUDA_CHECK(cudaMalloc(&common_rtx.dev_color, new_extent.width * new_extent.height * sizeof(float4)));

        if (common_rtx.dev_traced != 0)
                CUDA_CHECK(cudaFree((void *) common_rtx.dev_traced));

        common_rtx.dev_traced = (CUdeviceptr) cuda::alloc <uint32_t> (new_extent.width * new_extent.height);
        common_rtx.traced.resize(new_extent.width * new_extent.height * sizeof(uint32_t));

        // Send resize events
        sparse_gi.resize_queue.push(new_extent);
        mamba->resize_queue.push(new_extent);

        // Allocate Sobel filter output image
        sobel.output = ImageData {
                *phdev, *device,
                vk::Format::eR32Sfloat,
                vk::Extent2D { new_extent.width, new_extent.height },
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                vk::ImageAspectFlagBits::eColor
        };

        // Allocate staging buffer for querying
        index_staging_buffer = BufferData {
                *phdev, *device,
                new_extent.width * new_extent.height * sizeof(uint32_t),
                vk::BufferUsageFlagBits::eTransferDst
                        | vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible
                        | vk::MemoryPropertyFlagBits::eHostCoherent
        };

        // If needed, recreate framebuffer and rebind descriptor sets
        if (!render_state.initialized)
                return;

        // TODO: put the framebuffer code into a smaller struct
        // Recreate G-buffer framebuffer
        std::vector <vk::ImageView> attachment_views {
                *framebuffer_images->position.view,
                *framebuffer_images->normal.view,
                *framebuffer_images->uv.view,
                *framebuffer_images->material_index.view,
                *framebuffer_images->depth_buffer.view
        };

        vk::FramebufferCreateInfo fb_info {
                vk::FramebufferCreateFlags {},
                *gbuffer_render_pass,
                attachment_views,
                new_extent.width, new_extent.height, 1
        };

        gbuffer_fb = vk::raii::Framebuffer {*device, fb_info};

        // Resize viewport framebuffer
        std::vector <vk::ImageView> viewport_attachment_views {
                *framebuffer_images->viewport.view,
                *framebuffer_images->depth_buffer.view,
        };

        vk::FramebufferCreateInfo viewport_fb_info {
                vk::FramebufferCreateFlags {},
                *present_render_pass,
                viewport_attachment_views,
                new_extent.width, new_extent.height, 1
        };

        viewport_fb = vk::raii::Framebuffer {*device, viewport_fb_info};

        // Bind image to descriptor set
        std::array <vk::DescriptorImageInfo, 2> sobel_dset_image_infos {
                vk::DescriptorImageInfo {
                        nullptr,
                        *framebuffer_images->material_index.view,
                        vk::ImageLayout::eGeneral
                },

                vk::DescriptorImageInfo {
                        nullptr,
                        *sobel.output.view,
                        vk::ImageLayout::eGeneral
                },
        };

        std::array <vk::WriteDescriptorSet, 2> sobel_dset_writes {
                vk::WriteDescriptorSet {
                        *sobel.dset,
                        0, 0,
                        vk::DescriptorType::eStorageImage,
                        sobel_dset_image_infos[0],
                },

                vk::WriteDescriptorSet {
                        *sobel.dset,
                        1, 0,
                        vk::DescriptorType::eStorageImage,
                        sobel_dset_image_infos[1],
                },
        };

        device->updateDescriptorSets(sobel_dset_writes, nullptr);

        bind_ds(*device, triangulation.dset, framebuffer_images->position_sampler, framebuffer_images->position, 0);
        bind_ds(*device, triangulation.dset, framebuffer_images->normal_sampler, framebuffer_images->normal, 1);
        bind_ds(*device, triangulation.dset, framebuffer_images->material_index_sampler, framebuffer_images->material_index, 2);
        bind_ds(*device, triangulation.dset, sobel.output_sampler, sobel.output, 3);

        bind_ds(*device, normal.dset, framebuffer_images->normal_sampler, framebuffer_images->normal, 0);
        bind_ds(*device, uv.dset, framebuffer_images->uv_sampler, framebuffer_images->uv, 0);

        bind_ds(*device, highlight.dset,
                framebuffer_images->material_index_sampler,
                framebuffer_images->material_index, 0);
}

// Constructing framebuffer resources
FramebufferResources::FramebufferResources(const vk::raii::PhysicalDevice &phdev,
                const vk::raii::Device &device,
                const vk::Extent2D &extent_)
                : extent(extent_)
{
        static vk::Format formats[] = {
                vk::Format::eR32G32B32A32Sfloat, // Viewport
                vk::Format::eR32G32B32A32Sfloat, // Position
                vk::Format::eR32G32B32A32Sfloat, // Normal
                vk::Format::eR32G32B32A32Sfloat, // UV
                vk::Format::eR32Sint,            // Material index
        };

        // Other image propreties
        static vk::MemoryPropertyFlags mem_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        static vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
        static vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
        static vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment
                | vk::ImageUsageFlagBits::eTransferSrc
                | vk::ImageUsageFlagBits::eTransferDst
                | vk::ImageUsageFlagBits::eStorage;

        // Allocate viewport image
        viewport = ImageData {
                phdev, device,
                formats[0], extent, tiling,
                usage, mem_flags, aspect
        };

        position = ImageData {
                phdev, device,
                formats[1], extent, tiling,
                usage, mem_flags, aspect, true
        };

        normal = ImageData {
                phdev, device,
                formats[2], extent, tiling,
                usage, mem_flags, aspect, true
        };

        uv = ImageData {
                phdev, device,
                formats[3], extent, tiling,
                usage, mem_flags, aspect, true
        };

        material_index = ImageData {
                phdev, device,
                formats[4], extent, tiling,
                usage, mem_flags, aspect, true
        };

        depth_buffer = DepthBuffer {
                phdev, device,
                vk::Format::eD32Sfloat, extent
        };

        // Import into CUDA
        KOBRA_LOG_FUNC(Log::OK) << "Importing Vulkan textures into CUDA\n";

        cudaChannelFormatDesc channel_desc_f32 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        cudaChannelFormatDesc channel_desc_i32 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);

        import_vulkan_texture(device,
                position,
                cu_position_texture,
                cu_position_surface,
                channel_desc_f32);

        import_vulkan_texture(device,
                normal,
                cu_normal_texture,
                cu_normal_surface,
                channel_desc_f32);

        import_vulkan_texture(device,
                uv,
                cu_uv_texture,
                cu_uv_surface,
                channel_desc_f32);

        import_vulkan_texture(device,
                material_index,
                cu_material_index_texture,
                cu_material_index_surface,
                channel_desc_i32);

        // TODO: free old resources if needed...
        // TODO: cuda free the surfaces in the destructor

        // Allocate Vulkan samplers
        position_sampler = make_continuous_sampler(device);
        normal_sampler = make_continuous_sampler(device);
        uv_sampler = make_continuous_sampler(device);

        material_index_sampler = vk::raii::Sampler {
                device,
                vk::SamplerCreateInfo {
                        vk::SamplerCreateFlags {},
                        vk::Filter::eNearest,
                        vk::Filter::eNearest,
                        vk::SamplerMipmapMode::eNearest,
                        vk::SamplerAddressMode::eClampToEdge,
                        vk::SamplerAddressMode::eClampToEdge,
                        vk::SamplerAddressMode::eClampToEdge,
                        0.0f, VK_FALSE, 1.0f,
                        VK_FALSE, vk::CompareOp::eNever,
                        0.0f, 0.0f,
                        vk::BorderColor::eFloatOpaqueWhite,
                        VK_FALSE
                }
        };
}
