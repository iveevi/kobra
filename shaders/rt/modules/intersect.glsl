// Intersection
struct Intersection {
	float	time;
	vec3	normal;

	Material mat;
};

float b1;
float b2;

float _intersect_t(Triangle t, Ray r)
{
	vec3 e1 = t.v2 - t.v1;
	vec3 e2 = t.v3 - t.v1;
	vec3 s1 = cross(r.direction, e2);
	float divisor = dot(s1, e1);
	if (divisor == 0.0)
		return -1.0;
	vec3 s = r.origin - t.v1;
	float inv_divisor = 1.0 / divisor;
	b1 = dot(s, s1) * inv_divisor;
	if (b1 < 0.0 || b1 > 1.0)
		return -1.0;
	vec3 s2 = cross(s, e1);
	b2 = dot(r.direction, s2) * inv_divisor;
	if (b2 < 0.0 || b1 + b2 > 1.0)
		return -1.0;
	float time = dot(e2, s2) * inv_divisor;
	return time;
}

// Sphere-ray intersection
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

Intersection intersect_shape(Ray r, Triangle t)
{
	// Get intersection time
	float time = _intersect_t(t, r);
	vec3 n = vec3(0.0);

	if (time < 0.0)
		return Intersection(-1.0, n, mat_default());

	// Calculate the normal
	vec3 e1 = t.v2 - t.v1;
	vec3 e2 = t.v3 - t.v1;
	n = cross(e1, e2);
	n = normalize(n);

	// Negate normal if in the same direction as the ray
	if (dot(n, r.direction) > 0.0)
		n = -n;

	return Intersection(time, n, mat_default());
}

Intersection intersect_shape(Ray r, Sphere s)
{
	float t = _intersect_t(s, r);
	vec3 n = vec3(0, 0, 0);

	// If no, intersection, dont bother with normal
	if (t < 0.0)
		return Intersection(t, n, mat_default());

	// Calculate the normal
	n = normalize(r.origin + r.direction * t - s.center);

	return Intersection(t, n, mat_default());
}

Intersection ray_sphere_intersect(Ray ray, uint a, uint d)
{
	vec3 c = vertices.data[2 * a].xyz;
	float r = vertices.data[2 * a].w;

	Sphere s = Sphere(c, r);

	// Get intersection
	Intersection it = intersect_shape(ray, s);

	// If intersection is valid, compute material
	if (it.time > 0.0) {
		// TODO: function to do mat_at with texture coordinates
		// Get uv coordinates
		vec2 uv = vec2(0.0);
		uv.x = atan(ray.direction.x, ray.direction.z) / (2.0 * PI) + 0.5;
		uv.y = asin(ray.direction.y) / PI + 0.5;

		// Get the color
		// TODO: reuse from mesh
		it.mat.albedo = texture(s2_albedo[0], uv).rgb;

		// Get material index at the second element
		it.mat = mat_at(d, uv);
	}

	return it;
}

Intersection ray_intersect(Ray ray, uint index)
{
	float ia = triangles.data[index].x;
	float ib = triangles.data[index].y;
	float ic = triangles.data[index].z;
	float id = triangles.data[index].w;

	uint a = floatBitsToUint(ia);
	uint b = floatBitsToUint(ib);
	uint c = floatBitsToUint(ic);
	uint d = floatBitsToUint(id);

	// TODO: if a == b == c, then its a sphere with vertex at a and radius d
	if (a == b && b == c)
		return ray_sphere_intersect(ray, a, d);

	// TODO: macro for fixed width vertices
	vec3 v1 = vertices.data[2 * a].xyz;
	vec3 v2 = vertices.data[2 * b].xyz;
	vec3 v3 = vertices.data[2 * c].xyz;

	Triangle triangle = Triangle(v1, v2, v3);

	// Get intersection
	Intersection it = intersect_shape(ray, triangle);

	// If intersection is valid, compute material
	if (it.time > 0.0) {
		// Get texture coordinates
		vec2 t1 = vertices.data[2 * a + 1].xy;
		vec2 t2 = vertices.data[2 * b + 1].xy;
		vec2 t3 = vertices.data[2 * c + 1].xy;

		// Interpolate texture coordinates
		vec2 tex_coord = t1 * (1 - b1 - b2) + t2 * b1 + t3 * b2;
		tex_coord.y = 1.0 - tex_coord.y;

		// Transfer albedo
		it.mat = mat_at(d, tex_coord);

		// Transfer normal
		if (it.mat.has_normal < 0.5) {
			vec3 n = texture(s2_normals[d], tex_coord).rgb;
			n = 2 * n - 1.0;

			vec3 world_normal = normalize(it.normal);

			vec3 e1 = v2 - v1;
			vec3 e2 = v3 - v1;

			vec2 uv1 = t2 - t1;
			vec2 uv2 = t3 - t1;

			float r = 1.0 / (uv1.x * uv2.y - uv2.x * uv1.y);
			vec3 tangent = normalize(e1 * uv2.y - e2 * uv1.y) * r;
			vec3 bitangent = normalize(e2 * uv1.x - e1 * uv2.x) * r;

			mat3 tbn = mat3(tangent, bitangent, world_normal);

			it.normal = normalize(tbn * n);
		}
	}

	return it;
}

