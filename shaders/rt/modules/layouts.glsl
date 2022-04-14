layout (set = 0, binding = MESH_BINDING_PIXELS, std430) buffer Pixels
{
	uint pixels[];
} frame;

layout (set = 0, binding = MESH_BINDING_VERTICES, std430) buffer Vertices
{
	vec4 data[];
} vertices;

layout (set = 0, binding = MESH_BINDING_TRIANGLES, std430) buffer Triangles
{
	vec4 data[];
} triangles;

// Mesh transforms
// TODO: is this even needed? its too slow to compute every frame
layout (set = 0, binding = MESH_BINDING_TRANSFORMS, std430) buffer Transforms
{
	mat4 data[];
} transforms;

// Acceleration structure
layout (set = 0, binding = MESH_BINDING_BVH, std430) buffer BVH
{
	vec4 data[];
} bvh;

// Materials
layout (set = 0, binding = MESH_BINDING_MATERIALS, std430) buffer Materials
{
	vec4 data[];
} materials;

// Lights
layout (set = 0, binding = MESH_BINDING_LIGHTS, std430) buffer Lights
{
	vec4 data[];
} lights;

// Light indices
layout (set = 0, binding = MESH_BINDING_LIGHT_INDICES, std430) buffer LightIndices
{
	uint data[];
} light_indices;

// Textures
layout (set = 0, binding = MESH_BINDING_ALBEDOS)
uniform sampler2D s2_albedo[MAX_TEXTURES];

layout (set = 0, binding = MESH_BINDING_NORMAL_MAPS)
uniform sampler2D s2_normals[MAX_TEXTURES];

layout (set = 0, binding = MESH_BINDING_ENVIRONMENT)
uniform sampler2D s2_environment;

// Push constants
layout (push_constant) uniform PushConstants
{
	// Viewport
	uint	width;
	uint	height;
	uint	xoffset;
	uint	yoffset;

	// Size variables
	uint	triangles;
	uint	lights;
	uint	samples_per_pixel;
	uint	samples_per_light;

	// Other options
	uint	accumulate;	// TODO: replace with just present
	uint	present;
	uint	total;

	// Camera
	vec3	camera_position;
	vec3	camera_forward;
	vec3	camera_up;
	vec3	camera_right;

	// scale, aspect
	vec4	properties;
} pc;
