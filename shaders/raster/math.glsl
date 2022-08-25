#ifndef KOBRA_SHADERS_MATH_H_
#define KOBRA_SHADERS_MATH_H_

// Rotate a vector to orient it along a given direction
vec3 rotate(vec3 s, vec3 n)
{
	vec3 w = n;
	vec3 a = vec3(0.0f, 1.0f, 0.0f);

	if (dot(w, a) > 0.999f)
		a = vec3(0.0f, 0.0f, 1.0f);

	vec3 u = normalize(cross(w, a));
	vec3 v = normalize(cross(w, u));

	return u * s.x + v * s.y + w * s.z;
}

#endif
