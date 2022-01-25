#ifndef OBJECT_H_
#define OBJECT_H_

// Linear algebra
#include <glm/glm.hpp>

// Aliases
using glm::vec3;
using glm::mat4;

// Ray object
struct Ray {
	vec3 origin;
	vec3 direction;
};

// NOTE: shapes/objects are represented as support functions
struct Object {
        virtual vec3 support(const vec3 &) const = 0;

        // TODO: return float distance
	virtual bool intersect(const Ray &, vec3 &) const = 0;
};

// Sphere objects
struct Sphere : Object {
        vec3 center;
        float radius;

        Sphere(const vec3 &c, float r) : center(c), radius(r) {}

        vec3 support(const vec3 &d) const override {
                return center + radius * glm::normalize(d);
        }

	bool intersect(const Ray &ray, vec3 &pt) const override {
		vec3 oc = ray.origin - center;
		float a = glm::dot(ray.direction, ray.direction);
		float b = 2.0f * glm::dot(oc, ray.direction);
		float c = glm::dot(oc, oc) - radius * radius;
		float discriminant = b * b - 4.0f * a * c;
		if (discriminant < 0.0f) {
			return false;
		}

		float t = (-b - glm::sqrt(discriminant)) / (2.0f * a);
		if (t < 0.0f) {
			t = (-b + glm::sqrt(discriminant)) / (2.0f * a);
		}

		pt = ray.origin + ray.direction * t;
		return true;
	}
};

#endif
