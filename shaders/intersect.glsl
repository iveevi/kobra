// Intersection structure
struct Intersection {
	float	time;
	vec3	normal;
	vec3	color;
	float	shading;	// Shading type
};

float _intersect_t(Sphere s, Ray r)
{
	vec3 oc = r.origin - s.center;
	float a = dot(r.direction, r.direction);
	float b = 2.0 * dot(oc, r.direction);
	float c = dot(oc, oc) - s.radius * s.radius;
	float d = b * b - 4.0 * a * c;

	if (d < 0.0)
		return -1.0;
	
	float t1 = (-b - sqrt(d)) / (2.0 * a);
	float t2 = (-b + sqrt(d)) / (2.0 * a);

	return min(t1, t2);
}

// Return full information
Intersection intersect_shape(Ray r, Sphere s)
{
	float t = _intersect_t(s, r);
	vec3 n = vec3(0, 0, 0);

	// If no, intersection, dont bother with normal
	if (t < 0.0)
		return Intersection(t, n, vec3(0.0), SHADING_TYPE_NONE);

	// Calculate the normal
	n = normalize(r.origin + r.direction * t - s.center);

	return Intersection(t, n, vec3(0.0), SHADING_TYPE_NONE);
}
