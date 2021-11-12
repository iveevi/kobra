#version 330 core

layout (location = 0) in vec3 vpos;

out vec3 tex_coords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
	tex_coords= vpos;
	vec4 pos = projection * view * vec4(vpos, 1.0);
	gl_Position = pos.xyww;
}
