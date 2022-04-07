const vec4	hl_color = vec4(1.0, 1.0, 0.0, 1.0);
const float	hl_factor = 0.5;

#define HL_OUT()					\
	if (material.hightlight > 0)			\
		fragment = mix(hl_color, fragment, hl_factor);
