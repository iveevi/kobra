#version 450

// Inputs
layout (location = 0) in vec3 fcolor;
layout (location = 1) in vec2 fpos;

layout (binding = 0) uniform sampler2D tex;

// Outputs
layout (location = 0) out vec4 color;

// Main function
void main()
{
	vec4 a_color = texture(tex, fpos);
	color = vec4(1.0, 0.0, 1.0, 1.0);
	color = mix(color, a_color, 0.5);
}
