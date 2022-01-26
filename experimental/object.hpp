#ifndef OBJECT_H_
#define OBJECT_H_

// Linear algebra
#include <glm/glm.hpp>

// Engine headers
#include "ray.hpp"

// Aliases
using glm::vec3;
using glm::mat4;

// NOTE: shapes/objects are represented as support functions
struct Object {
        // Position
        vec3 position;
        
        // TODO: Add support for rotation and scaling

        // Constructors
        Object() : position(0.0f) {}
        Object(vec3 pos) : position(pos) {}
};

// Separate interface for those which can be rendered
struct Renderable {
        // TODO: is this necessary?
        virtual vec3 support(const vec3 &) const = 0;

        // TODO: return float distance
	virtual bool intersect(const Ray &, vec3 &) const = 0;

        // Get the normal at the given point
        virtual vec3 normal(const vec3 &) const = 0;
};

// Sphere objects
struct Sphere : public virtual Object, public virtual Renderable {
        float radius;

        Sphere(const vec3 &c, float r) : Object(c), radius(r) {}

        vec3 support(const vec3 &d) const override {
                return position + radius * glm::normalize(d);
        }

	bool intersect(const Ray &ray, vec3 &pt) const override {
		vec3 oc = ray.origin - position;
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

        vec3 normal(const vec3 &pt) const override {
                return glm::normalize(pt - position);
        }
};

#endif
