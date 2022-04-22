#include "../../../include/types.hpp"

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
	vec3 c = vertices.data[VERTEX_STRIDE * a].xyz;
	float r = vertices.data[VERTEX_STRIDE * a].w;

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
		it.mat.albedo = texture(s2_albedo[0], uv).rgb;

		// Get material index at the second element
		it.mat = mat_at(d, uv);
	}

	return it;
}

Intersection ray_intersect(Ray ray, uint index)
{
	uvec4 i = floatBitsToUint(triangles.data[index]);

	// TODO: if a == b == c, then its a sphere with vertex at a and radius d
	if (i.x == i.y && i.y == i.z)
		return ray_sphere_intersect(ray, i.x, i.w);

	vec3 v1 = vertices.data[VERTEX_STRIDE * i.x].xyz;
	vec3 v2 = vertices.data[VERTEX_STRIDE * i.y].xyz;
	vec3 v3 = vertices.data[VERTEX_STRIDE * i.z].xyz;

	Triangle triangle = Triangle(v1, v2, v3);

	// Get intersection
	Intersection it = intersect_shape(ray, triangle);

	// If intersection is valid, compute material
	if (it.time > 0.0) {
		// Get texture coordinates
		vec2 t1 = vertices.data[VERTEX_STRIDE * i.x + 1].xy;
		vec2 t2 = vertices.data[VERTEX_STRIDE * i.y + 1].xy;
		vec2 t3 = vertices.data[VERTEX_STRIDE * i.z + 1].xy;

		// Interpolate texture coordinates
		vec2 tex_coord = t1 * (1 - b1 - b2) + t2 * b1 + t3 * b2;
		tex_coord.y = 1.0 - tex_coord.y;

		// Transfer albedo
		it.mat = mat_at(i.w, tex_coord);

		// Transfer normal
		vec3 n1 = vertices.data[VERTEX_STRIDE * i.x + 2].xyz;
		vec3 n2 = vertices.data[VERTEX_STRIDE * i.y + 2].xyz;
		vec3 n3 = vertices.data[VERTEX_STRIDE * i.z + 2].xyz;

		// Interpolate vertex normal
		vec3 n = n1 * (1 - b1 - b2) + n2 * b1 + n3 * b2;

		it.normal = normalize(n);

		// Transfer normal
		if (it.mat.has_normal < 0.5) {
			vec3 n = texture(s2_normals[i.w], tex_coord).xyz;
			n = 2 * n - 1;

			// Get (interpolated) tangent and bitangent
			vec3 t1 = vertices.data[VERTEX_STRIDE * i.x + 3].xyz;
			vec3 t2 = vertices.data[VERTEX_STRIDE * i.y + 3].xyz;
			vec3 t3 = vertices.data[VERTEX_STRIDE * i.z + 3].xyz;

			vec3 t = t1 * (1 - b1 - b2) + t2 * b1 + t3 * b2;

			vec3 bit1 = vertices.data[VERTEX_STRIDE * i.x + 4].xyz;
			vec3 bit2 = vertices.data[VERTEX_STRIDE * i.y + 4].xyz;
			vec3 bit3 = vertices.data[VERTEX_STRIDE * i.z + 4].xyz;

			vec3 b = bit1 * (1 - b1 - b2) + bit2 * b1 + bit3 * b2;

			// TBN matrix
			mat3 tbn = mat3(t, b, it.normal);

			// Transform normal
			it.normal = normalize(tbn * n);
		}
	}

	return it;
}

