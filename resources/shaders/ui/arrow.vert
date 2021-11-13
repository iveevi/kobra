layout (location = 0) in vec3 v_pos;

uniform mat4 model;
uniform mat4 view;
// uniform mat4 projection;

void main()
{
	gl_Position = view * model * vec4(v_pos, 1.0);
}