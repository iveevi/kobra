#version 450

// Modules
#include "bindings.h"
#include "brdf.glsl"
#include "highlight.glsl"
#include "io_set.glsl"
#include "light_set.glsl"
#include "random.glsl"

vec3 sample_area_light(AreaLight al, inout vec3 seed)
{
	vec3 rand = random3(seed);
	float u = fract(rand.x);
	float v = fract(rand.y);
	return al.a + u * al.ab + v * al.ac;
}

vec3 brdf(Material mat, vec3 n, vec3 wi, vec3 wo)
{
	vec3 diffuse = mat.diffuse/PI;
		

	/* TODO: choose shading model
	float k = (mat.shininess + 2.0f)/(2.0f * PI);
	vec3 reflect = reflect(wo, n);
	float dot = pow(abs(dot(wi, reflect)), mat.shininess);
	vec3 specular = mat.specular * k * dot; */

	vec3 specular = ggx_brdf(mat, n, wi, wo);

	return diffuse + specular;
}

void main()
{
	// Get the normal
	vec3 n = normalize(normal);

	// Copy texture
	Material m = mat;
	
	// Resolve textures
	if (mat.has_albedo > 0.5)
		m.diffuse = texture(albedo_map, tex_coord).rgb;

	if (mat.has_normal > 0.5) {
		n = texture(normal_map, tex_coord).rgb;
		n = 2 * n - 1;
		n = normalize(tbn * n);
	}
	
	// First check if the object is emissive
	if (is_type(mat.type, SHADING_EMISSIVE)) {
		fragment = vec4(mat.diffuse, 1.0);
		HL_OUT();
		return;
	}

	// Sum up the light contributions
	vec3 color = vec3(0.0);

	// Area lights
	vec3 seed = fract(vec3(time * n - position));
	vec3 wo = normalize(view_pos - position);

	for (int i = 0; i < n_area_lights; i++) {
		AreaLight light = area_lights[i];

		int N = 4;
		for (int i = 0; i < N; i++) {
			vec3 lpos = sample_area_light(light, seed);
			vec3 wi = normalize(lpos - position);
			float R = length(lpos - position);

			// BRDF value
			vec3 f = brdf(m, n, wi, wo) * max(dot(wi, n), 0.0);

			// Area light normal
			vec3 Lc = cross(light.ab, light.ac);
			vec3 Ln = normalize(Lc);
			float area = length(Lc);
			float ldot = abs(dot(wi, Ln));

			// Contribution
			color += light.intensity * f * ldot * area/(R * R);
		}

		color /= float(N);
	}

	// Gamma correction
	color = clamp(color, 0.0, 1.0);
	color = pow(color, vec3(1.0/2.2));

	// Output
	fragment = vec4(color, 1.0);
	HL_OUT();
}
