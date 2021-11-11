#include lighting

in vec3 normal;
in vec3 frag_pos;

out vec4 frag_color;

// TODO: put these into a material struct
uniform vec3 color;
uniform vec3 view_pos;

#define NLIGHTS 4

uniform PointLight point_lights[NLIGHTS];
uniform int npoint_lights;

void main()
{
	vec3 result = vec3(0.0, 0.0, 0.0);
	for (int i = 0; i < npoint_lights; i++)
		result += point_light_contr(point_lights[i], color, frag_pos, view_pos, normal);
	frag_color = vec4(result, 1.0);
}
