#version 450

// Typical vertex shader
layout (location = 0) in vec3 position;
// layout (location = 1) in vec3 normal;
// layout (location = 2) in vec2 tex_coord;
// layout (location = 3) in vec3 tangent;
// layout (location = 4) in vec3 bitangent;

// Push constants
layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;

	uint id;
};

// Only output is id
layout (location = 0) out uint out_id;

void main()
{
	gl_Position = proj * view * model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w)/2.0;
	out_id = id;
}
