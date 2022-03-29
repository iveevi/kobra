#version 450

// Typical vertex shader
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_coord;

// Material structure
// TODO: header
struct Material {
	// Material properties
	vec3 albedo;
	float reflectance;
	float refractance;
	float extinction;
};

// MVP matrix as push constant
layout (push_constant) uniform PushConstants
{
	// Transform matrices
	mat4 model;
	mat4 view;
	mat4 proj;

	// Material properties
	Material material;
};

// Out color
layout (location = 0) out vec4 color;
layout (location = 1) out vec3 position_out;
layout (location = 2) out vec3 normal_out;
layout (location = 3) out vec2 tex_coord_out;
layout (location = 4) out Material material_out;

void main()
{
	// Transform vertex position by model, view and projection matrices
	gl_Position = proj * view * model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;

	// Output necessary info
	color		= vec4(1.0, 0.0, 0.0, 1.0);
	position_out	= (model * vec4(position, 1.0)).xyz;
	normal_out	= normal;
	tex_coord_out	= tex_coord;
	material_out	= material;
}
