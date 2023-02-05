#version 450

// Typical vertex shader
layout (location = 0) in vec3 position;

// Push constants
layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
	uvec2 id;
};

// Only output is id
layout (location = 0) out uvec2 out_id;

void main()
{
	gl_Position = proj * view * model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w)/2.0;
	out_id = id;
}
