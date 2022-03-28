// Bounding box
struct BoundingBox {
	vec3 pmin;
	vec3 pmax;
};

// Intersect bounding box
float intersect_box(Ray ray, BoundingBox box)
{
	float tmin = (box.pmin.x - ray.origin.x) / ray.direction.x;
	float tmax = (box.pmax.x - ray.origin.x) / ray.direction.x;

	// TODO: swap function?
	if (tmin > tmax) {
		float tmp = tmin;
		tmin = tmax;
		tmax = tmp;
	}

	float tymin = (box.pmin.y - ray.origin.y) / ray.direction.y;
	float tymax = (box.pmax.y - ray.origin.y) / ray.direction.y;

	if (tymin > tymax) {
		float tmp = tymin;
		tymin = tymax;
		tymax = tmp;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return -1.0;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (box.pmin.z - ray.origin.z) / ray.direction.z;
	float tzmax = (box.pmax.z - ray.origin.z) / ray.direction.z;

	if (tzmin > tzmax) {
		float tmp = tzmin;
		tzmin = tzmax;
		tzmax = tmp;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return -1.0;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	return tmin;
}

// Check if point is in bounding box
bool in_box(vec3 point, BoundingBox box)
{
	return (point.x >= box.pmin.x && point.x <= box.pmax.x &&
		point.y >= box.pmin.y && point.y <= box.pmax.y &&
		point.z >= box.pmin.z && point.z <= box.pmax.z);
}
