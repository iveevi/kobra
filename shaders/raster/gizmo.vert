#version 450

// Only use position
layout(location = 0) in vec3 position;

// Push constants
layout(push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;

	vec3 color;
};

// Output
layout(location = 0) out vec3 out_color;

void main()
{
	gl_Position = proj * view * model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w)/2.0;
	out_color = color;
}
