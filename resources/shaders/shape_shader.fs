#version 330 core

out vec4 color;

uniform vec3 shape_color;

void main()
{
	color = vec4(shape_color, 1.0);
}
