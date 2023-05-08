
// Vulkan headers
#include <vulkan/vulkan_format_traits.hpp>

// Editor headers
#include "editor_viewport.cuh"

// Engien headers
#include "include/cuda/error.cuh"
#include "include/cuda/interop.cuh"
#include "include/daemons/mesh.hpp"

// Environment map
void load_environment_map(EnvironmentMap *em, kobra::TextureLoader *loader, const std::filesystem::path &path)
{
 //        kobra::RawImage img = kobra::load_texture(path);
	// 
 //        // TODO: submit now instead...
 //        // Queue to submit commands to
 //        vk::raii::Queue queue { *context.device, 0, 0 };
	//
 //        // Temporary command buffer
 //        auto cmd = make_command_buffer(*context.device, *context.command_pool);
	//
 //        // Create the image
 //        vk::Extent2D extent {
 //                static_cast <uint32_t> (img.width),
 //                static_cast <uint32_t> (img.height)
 //        };
	//
 //        // Select format
 //        vk::Format format = vk::Format::eR8G8B8A8Unorm;
 //        if (img.type == RawImage::RGBA_32_F)
 //                format = vk::Format::eR32G32B32A32Sfloat;
	//
 //        em->image = ImageData(
 //                *context.phdev, *context.device,
 //                format, extent,
 //                vk::ImageTiling::eOptimal,
 //                vk::ImageUsageFlagBits::eSampled
 //                        | vk::ImageUsageFlagBits::eTransferDst
 //                        | vk::ImageUsageFlagBits::eTransferSrc,
 //                vk::MemoryPropertyFlagBits::eDeviceLocal,
 //                vk::ImageAspectFlagBits::eColor,
 //                true
 //        );
	//
 //        // Copy the image data into a staging buffer
 //        vk::DeviceSize size = img.width * img.height * vk::blockSize(em->image.format);
	//
 //        BufferData buffer {
 //                *context.phdev, *context.device, size,
 //                vk::BufferUsageFlagBits::eTransferSrc,
 //                vk::MemoryPropertyFlagBits::eHostVisible
 //                        | vk::MemoryPropertyFlagBits::eHostCoherent
 //        };
	//
 //        // Copy the data
 //        buffer.upload(img.data);
	//
 //        {
 //                cmd.begin({});
 //                em->image.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);
	//
 //                // Copy the buffer to the image
 //                copy_data_to_image(cmd,
 //                        buffer.buffer, em->image.image,
 //                        em->image.format, img.width, img.height
 //                );
	//
 //                // TODO: transition_image_layout should go to the detail namespace...
 //                em->image.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
 //                cmd.end();
 //        }
	//
 //        // Submit the command buffer
 //        queue.submit(
 //                vk::SubmitInfo {
 //                        0, nullptr, nullptr,
 //                        1, &*cmd
 //                },
 //                nullptr
 //        );
	//
 //        // Wait
 //        queue.waitIdle();
	//
 //        // Import into CUDA
 //        // TODO: check if 8u or 32f
 //        if (img.type == RawImage::RGBA_32_F)
 //                em->texture = kobra::cuda::import_vulkan_texture_32f(*context.device, em->image);
 //        else
 //                em->texture = kobra::cuda::import_vulkan_texture_8u(*context.device, em->image);

        const ImageData &img = loader->load_texture(path);
        em->texture = kobra::cuda::import_vulkan_texture(*loader->m_device.device, img);

        // Mark as valid
        em->valid = true;

        std::cout << "Environment map loaded for editor viewport" << std::endl;
}

// Constructor
EditorViewport::EditorViewport
                        (const Context &context,
                        const std::shared_ptr <amadeus::Accelerator> &_system,
                        const std::shared_ptr <daemons::MeshDaemon> &_mesh_memory)
                : system(_system),
                mesh_memory(_mesh_memory),
                phdev(context.phdev),
                device(context.device),
                descriptor_pool(context.descriptor_pool),
                command_pool(context.command_pool),
                texture_loader(context.texture_loader)
{
        common_rtx.timer.start();

        path_tracer.dev_traced = 0;
        path_tracer.launch_params.color = 0;
        path_tracer.depth = 2;

        // amadeus_path_tracer.depth = 2;

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
        // configure_amadeus_path_tracer(context);
        configure_path_tracer(context);

        render_state.initialized = true;
}

static void import_vulkan_texture
                (const vk::raii::Device &device,
                const ImageData &image,
                // cudaTextureObject_t &texture,
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
        
        // CUDA_CHECK(cudaCreateTextureObject(
        //         &texture,
        //         &res_desc, &tex_desc, nullptr
        // ));
}

void EditorViewport::resize(const vk::Extent2D &new_extent)
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
        framebuffer_images.viewport = ImageData {
                *phdev, *device,
                formats[0], new_extent, tiling,
                usage, mem_flags, aspect
        };

        framebuffer_images.position = ImageData {
                *phdev, *device,
                formats[1], new_extent, tiling,
                usage, mem_flags, aspect, true
        };

        framebuffer_images.normal = ImageData {
                *phdev, *device,
                formats[2], new_extent, tiling,
                usage, mem_flags, aspect, true
        };

        framebuffer_images.uv = ImageData {
                *phdev, *device,
                formats[3], new_extent, tiling,
                usage, mem_flags, aspect, true
        };

        framebuffer_images.material_index = ImageData {
                *phdev, *device,
                formats[4], new_extent, tiling,
                usage, mem_flags, aspect, true
        };

        depth_buffer = DepthBuffer {
                *phdev, *device,
                vk::Format::eD32Sfloat, new_extent
        };

        // Import into CUDA
        // TODO: free old resources if needed...
        KOBRA_LOG_FUNC(Log::OK) << "Importing Vulkan textures into CUDA\n";

        cudaChannelFormatDesc channel_desc_f32 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        // cudaChannelFormatDesc channel_desc_f32_rg = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
        cudaChannelFormatDesc channel_desc_i32 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);

        import_vulkan_texture(*device,
                framebuffer_images.position,
                framebuffer_images.cu_position_surface,
                channel_desc_f32);

        import_vulkan_texture(*device,
                framebuffer_images.normal,
                framebuffer_images.cu_normal_surface,
                channel_desc_f32);

        import_vulkan_texture(*device,
                framebuffer_images.uv,
                framebuffer_images.cu_uv_surface,
                channel_desc_f32);

        import_vulkan_texture(*device,
                framebuffer_images.material_index,
                framebuffer_images.cu_material_index_surface,
                channel_desc_i32);

        // Allocate resources for the G-buffer based path tracer
        if (path_tracer.launch_params.color != 0)
                CUDA_CHECK(cudaFree(path_tracer.launch_params.color));

        CUDA_CHECK(cudaMalloc(&path_tracer.launch_params.color,
                new_extent.width * new_extent.height * sizeof(float4)));
        
        if (path_tracer.dev_traced != 0)
                CUDA_CHECK(cudaFree((void *) path_tracer.dev_traced));

        CUDA_CHECK(cudaMalloc((void **) &path_tracer.dev_traced, new_extent.width * new_extent.height * sizeof(uint32_t)));

        path_tracer.traced.resize(new_extent.width * new_extent.height * sizeof(uint32_t));

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
        if (render_state.initialized) {
                // TODO: put the framebuffer code into a smaller function
                // Recreate G-buffer framebuffer
                std::vector <vk::ImageView> attachment_views {
                        *framebuffer_images.position.view,
                        *framebuffer_images.normal.view,
                        *framebuffer_images.uv.view,
                        *framebuffer_images.material_index.view,
                        *depth_buffer.view
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
                        *framebuffer_images.viewport.view,
                        *depth_buffer.view,
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
                                *framebuffer_images.material_index.view,
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
        
                bind_ds(*device, triangulation.dset, framebuffer_images.position_sampler, framebuffer_images.position, 0);
                bind_ds(*device, triangulation.dset, framebuffer_images.normal_sampler, framebuffer_images.normal, 1);
                bind_ds(*device, triangulation.dset, framebuffer_images.material_index_sampler, framebuffer_images.material_index, 2);
                bind_ds(*device, triangulation.dset, sobel.output_sampler, sobel.output, 3);

                bind_ds(*device, normal.dset, framebuffer_images.normal_sampler, framebuffer_images.normal, 0);
                bind_ds(*device, uv.dset, framebuffer_images.uv_sampler, framebuffer_images.uv, 0);
        
                bind_ds(*device, highlight.dset,
                        framebuffer_images.material_index_sampler,
                        framebuffer_images.material_index, 0);
        } else {
                // First time initialization

                // Create samplers for the framebuffer images
                framebuffer_images.position_sampler = make_continuous_sampler(*device);
                framebuffer_images.normal_sampler = make_continuous_sampler(*device);
                framebuffer_images.uv_sampler = make_continuous_sampler(*device);

                framebuffer_images.material_index_sampler = vk::raii::Sampler {
                        *device,
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

        // Update extent
        extent = new_extent;
}
