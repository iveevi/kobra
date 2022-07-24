#version 450

// Inputs
layout (location = 0) in vec3 in_color;
layout (location = 1) in vec2 in_uv;

// Outputs
layout(location = 0) out vec4 fragment;

// TODO: rename to rect.frag or something
// TODO: radius and thickness
layout (push_constant) uniform PushConstant {
	vec2 center;
	float width;
	float height;
	float radius;
	float thickness;
};

float sdf_round_box(in vec2 p, in vec2 b, in vec4 r)
{
	r.xy = (p.x > 0.0) ? r.xy : r.zw;
	r.x  = (p.y > 0.0) ? r.x : r.y;

	vec2 q = abs(p) - b + r.x;

	return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}

// Main function
void main()
{
	vec2 uv = (in_uv - center);

	float d = sdf_round_box(uv, vec2(width, height)/2, vec4(radius));
	if (d > 0.0f)
		discard;

	vec3 col = mix(in_color, vec3(1.0), 1.0 - smoothstep(0.0, thickness, abs(d)));
	fragment = vec4(col, 1.0);
}
