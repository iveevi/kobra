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

// Area lights
struct AreaLight {
	vec3 a;
	vec3 ab;
	vec3 ac; // d = a + ab + ac
	vec3 color;
	float power;
};

layout (set = 0, binding = MESH_BINDING_AREA_LIGHTS, std430) buffer AreaLights
{
	int count;

	AreaLight data[];
} area_lights;

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

	uint	skip;
	uint	xoffset;
	uint	yoffset;

	// Size variables
	uint	triangles;
	uint	lights;

	// Sample counts
	// TODO: make floats
	uint	samples_per_pixel;
	uint	samples_per_surface;
	uint	samples_per_light;

	// Other options
	uint	accumulate;	// TODO: replace with just present
	uint	present;
	uint	total;

	// Other variables
	float	time;

	// Camera
	vec3	camera_position;
	vec3	camera_forward;
	vec3	camera_up;
	vec3	camera_right;

	// scale, aspect
	vec4	properties;
} pc;
