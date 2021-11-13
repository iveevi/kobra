#include emit

layout (lines) in;
layout (line_strip, max_vertices = 10) out;

uniform mat4 projection;
uniform int start_mode;
uniform int end_mode;

void _draw_start(vec4 p1, vec4 p2)
{
	// TODO: do in separate functions
	if (start_mode > 0) {
		vec2 p_p1 = p1.xy;
		vec2 p_p2 = p2.xy;

		vec2 diff = normalize(p_p1 - p_p2);
		vec2 hori = vec2(1.0, 0.0);
		float angle = acos(dot(diff, hori));

		// TODO: set angle as a parameter
		float a1 = angle - 0.5;
		float a2 = angle + 0.5;

		_emit_line(p1, p1 - projection * vec4(cos(a1), sin(a1), 0.0, 0.0));
		_emit_line(p1, p1 - projection * vec4(cos(a2), sin(a2), 0.0, 0.0));
	}
}

void _draw_end(vec4 p1, vec4 p2)
{
	if (end_mode > 0) {
		vec2 p_p1 = p1.xy;
		vec2 p_p2 = p2.xy;

		vec2 diff = normalize(p_p1 - p_p2);
		vec2 hori = vec2(1.0, 0.0);
		float angle = acos(dot(diff, hori));

		float a1 = angle - 0.5;
		float a2 = angle + 0.5;

		_emit_line(p2, p2 + projection * vec4(cos(a1), sin(a1), 0.0, 0.0));
		_emit_line(p2, p2 + projection * vec4(cos(a2), sin(a2), 0.0, 0.0));
	}
}

void main()
{
	vec4 p1 = projection * gl_in[0].gl_Position;
	vec4 p2 = projection * gl_in[1].gl_Position;

	_emit_line(p1, p2);
	_draw_start(p1, p2);
	_draw_end(p1, p2);
}