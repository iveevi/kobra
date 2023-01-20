// Engine headers
#include "../../include/allocator.hpp"
#include "../../include/camera.hpp"
#include "../../include/layers/gizmo.hpp"
#include "../../include/vertex.hpp"

namespace kobra {

namespace layers {

// Push constants
struct PushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;

	alignas(16)
	glm::vec3 color;
};

// Create rasterizer for translate gizmo
static Renderable *make_translate_rasterizer(Context &context)
{
	Submesh cone_x = Submesh::cone();
	Submesh cone_y = Submesh::cone();
	Submesh cone_z = Submesh::cone();

	Submesh cylinder_x = Submesh::cylinder();
	Submesh cylinder_y = Submesh::cylinder();
	Submesh cylinder_z = Submesh::cylinder();

	// Transform each submesh appropriately
	Transform cone_transform;
	Transform cylinder_transform;

	cone_transform.scale = glm::vec3(0.3f, 0.5f, 0.3f);
	cylinder_transform.scale = glm::vec3(0.1f, 5.0f, 0.1f);
	
	// X
	cone_transform.position = {5.0f, 0, 0};
	cone_transform.rotation.z = -90;
	cylinder_transform.position = {2.5f, 0, 0};
	cylinder_transform.rotation.z = -90;

	Submesh::transform(cone_x, cone_transform);
	Submesh::transform(cylinder_x, cylinder_transform);

	// Y
	cone_transform.position = {0, 5.0f, 0};
	cone_transform.rotation.z = 0;
	cone_transform.rotation.x = 0;
	cylinder_transform.position = {0, 2.5f, 0};
	cylinder_transform.rotation.z = 0;
	cylinder_transform.rotation.x = 0;

	Submesh::transform(cone_y, cone_transform);
	Submesh::transform(cylinder_y, cylinder_transform);

	// Z
	cone_transform.position = {0, 0, 5.0f};
	cone_transform.rotation.x = 90;
	cone_transform.rotation.z = 0;
	cylinder_transform.position = {0, 0, 2.5f};
	cylinder_transform.rotation.x = 90;
	cylinder_transform.rotation.z = 0;

	Submesh::transform(cone_z, cone_transform);
	Submesh::transform(cylinder_z, cylinder_transform);

	// Sphere dot
	Submesh sphere = Submesh::sphere();

	Transform sphere_transform;
	sphere_transform.scale = {0.1f, 0.1f, 0.1f};

	Submesh::transform(sphere, sphere_transform);

	// Populate a mesh (TODO: turn in to a move operation)
	// TODO: store this mesh in the structure
	Mesh *mesh = new Mesh {{
		cone_x,
		cone_y,
		cone_z,
		cylinder_x,
		cylinder_y,
		cylinder_z,
		sphere
	}};

	// Create a rasterizer
	return new Renderable(context, mesh);
}

// Get model matrix for scaling gizmo
static Transform get_scale_model(const Transform &origin,
		const Camera &camera,
		const Transform &camera_transform)
{
	// TODO: scale the gizmo so it's always the same size on screen
	Transform model;

	float d = glm::length(camera_transform.position - origin.position);
	model.position = origin.position;
	model.scale = glm::vec3(d * 0.04f);

	return model;
}

// Create a gizmo layer
Gizmo Gizmo::make(Context &context)
{
	// Layer to return
	Gizmo layer;

	// Get extent
	layer.extent = context.extent;

	// First create a render pass
	auto eLoad = vk::AttachmentLoadOp::eLoad;
	layer.render_pass = make_render_pass(*context.device,
		{context.swapchain_format}, {eLoad},
		context.depth_format, {eLoad}
	);

	// Pipeline layout
	auto pc = vk::PushConstantRange {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(PushConstants)
	};

	layer.ppl = vk::raii::PipelineLayout {
		*context.device,
		{{}, {}, pc}
	};

	// Create the pipeline
	auto shaders = make_shader_modules(*context.device, {
		"bin/spv/gizmo_vert.spv",
		"bin/spv/gizmo_frag.spv"
	});

	auto vertex_binding = Vertex::vertex_binding();
	auto vertex_attributes = Vertex::vertex_attributes();

	GraphicsPipelineInfo grp_info {
		*context.device, layer.render_pass,
		std::move(shaders[0]), nullptr,
		std::move(shaders[1]), nullptr,
		vertex_binding, vertex_attributes,
		layer.ppl
	};

	// Gizmo will always appear on top of everything else
	grp_info.depth_test = false;
	grp_info.depth_write = false;

	layer.pipeline = make_graphics_pipeline(grp_info);

	// Create rasterizers for each gizmo
	layer.translate = make_translate_rasterizer(context);

	// Return
	return layer;
}

// Render the gizmo
// TODO: last 4 arguments into a struct
void Gizmo::render(Gizmo &layer, Type type, const Transform &origin,
		const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const Camera &camera,
		const Transform &transform,
		const RenderArea &ra)
{
	// Set render area
	ra.apply(cmd, layer.extent);

	// Begin render pass
	std::array <vk::ClearValue, 1> clear_values {
		vk::ClearColorValue {
			std::array <float, 4> {0.0f, 0.0f, 0.0f, 0.0f}
		},
	};

	cmd.beginRenderPass(
		vk::RenderPassBeginInfo {
			*layer.render_pass,
			*framebuffer,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				layer.extent
			},
			static_cast <uint32_t> (clear_values.size()),
			clear_values.data()
		},
		vk::SubpassContents::eInline
	);

	// Bind pipeline
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *layer.pipeline);
	
	// TODO: scale the gizmo so it's always the same size on screen
	Transform model = get_scale_model(origin, camera, transform);

	// Setup push constants
	PushConstants pc {
		.model = model.matrix(), // TODO: remove scale and rotation
		.view = camera.view_matrix(transform),
		.projection = camera.perspective_matrix()
	};

	if (type == Type::eTranslate) {
		std::vector <glm::vec3> colors {
			{1.0f, 0.0f, 0.0f},
			{0.0f, 1.0f, 0.0f},
			{0.0f, 0.0f, 1.0f},

			{1.0f, 0.0f, 0.0f},
			{0.0f, 1.0f, 0.0f},
			{0.0f, 0.0f, 1.0f},

			{1.0f, 1.0f, 1.0f}
		};

		for (size_t i = 0; i < colors.size(); i++) {
			pc.color = colors[i];
			
			cmd.pushConstants <PushConstants> (*layer.ppl,
				vk::ShaderStageFlagBits::eVertex,
				0, pc
			);

			cmd.bindVertexBuffers(0, *layer.translate->get_vertex_buffer(i).buffer, {0});
			cmd.bindIndexBuffer(*layer.translate->get_index_buffer(i).buffer,
				0, vk::IndexType::eUint32
			);

			cmd.drawIndexed(layer.translate->get_index_count(i), 1, 0, 0, 0);
		}
	}

	// End render pass
	cmd.endRenderPass();
}

// Closest distance between line segment and point
static float line_segment_to_point(const glm::vec2 &a, const glm::vec2 &b, const glm::vec2 &p)
{
	glm::vec2 ab = b - a;
	glm::vec2 ap = p - a;
	float t = glm::dot(ap, ab) / glm::dot(ab, ab);
	if (t < 0.0f)
		return glm::length(p - a);
	if (t > 1.0f)
		return glm::length(p - b);
	return glm::length(p - a - t * ab);
}

// Handle dragging
bool Gizmo::handle(Gizmo &layer, Type type, Transform &origin,
		const Camera &camera,
		const Transform &camera_transform,
		const RenderArea &ra,
		const glm::vec2 &mouse_pos,
		const glm::vec2 &mouse_delta,
		bool cache)
{
	static int cached_axis = -1;
		
	// Get camera projection matrix
	glm::mat4 proj = camera.perspective_matrix()
		* camera.view_matrix(camera_transform);
	
	// Project point
	int w = ra.max.x - ra.min.x;
	int h = ra.max.y - ra.min.y;
	auto projector = [&] (const glm::vec4 &p) {
		glm::vec4 pp = proj * p;
		pp /= pp.w;

		glm::vec2 o {pp.x, pp.y};
		o = o * 0.5f + 0.5f;
		o.y = 1.0f - o.y;

		return o * glm::vec2 {w, h} + ra.min;
	};

	// Get model matrix
	Transform model = get_scale_model(origin, camera, camera_transform);

	// Axial vectors
	glm::vec3 x {5.0f, 0.0f, 0.0f};
	glm::vec3 y {0.0f, 5.0f, 0.0f};
	glm::vec3 z {0.0f, 0.0f, 5.0f};

	// Get the four corners of the gizmo
	glm::vec4 p0 = {model.position, 1};
	glm::vec4 p1 = {model.apply(x), 1};
	glm::vec4 p2 = {model.apply(y), 1};
	glm::vec4 p3 = {model.apply(z), 1};

	// Project them
	glm::vec2 o0 = projector(p0);
	glm::vec2 o1 = projector(p1);
	glm::vec2 o2 = projector(p2);
	glm::vec2 o3 = projector(p3);

	// Const data
	const glm::vec2 corners[3] = {o1, o2, o3};

	const float scales[3] = {
		glm::length(o1 - o0),
		glm::length(o2 - o0),
		glm::length(o3 - o0)
	};

	// If not cached:
	if (!cache || cached_axis == -1) {
		// Get the closest distance from axes to mouse
		float d1 = line_segment_to_point(o0, o1, mouse_pos);
		float d2 = line_segment_to_point(o0, o2, mouse_pos);
		float d3 = line_segment_to_point(o0, o3, mouse_pos);

		// Check which axes are valid (per threshold)
		static const float threshold = 30.0f;
		
		std::vector <int> valid;
		if (d1 < threshold) valid.push_back(0);
		if (d2 < threshold) valid.push_back(1);
		if (d3 < threshold) valid.push_back(2);

		if (valid.empty()) {
			cached_axis = -1;
			return false;
		}

		// Get axis which is most "relevant"
		int amin = -1;
		float dmax = 0.0f;

		glm::vec2 nmpos = glm::normalize(mouse_pos);
		for (int i : valid) {
			float t = glm::dot(nmpos, glm::normalize(corners[i] - o0));
			if (std::abs(t) > dmax) {
				dmax = std::abs(t);
				amin = i;
			}
		}

		assert(amin != -1);
		cached_axis = amin;
	}

	// Translate
	int amin = cached_axis;
	glm::vec2 c = corners[amin];
	float t = glm::dot(glm::normalize(mouse_delta), glm::normalize(c - o0));
	origin.position[amin] += t/2.0f;

	return true;
}

}

}
