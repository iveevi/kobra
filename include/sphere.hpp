#ifndef SPHERE_H_
#define SPHERE_H_

// Standard headers
#include <optional>

// Engine headers
#include "renderable.hpp"
#include "object.hpp"

namespace kobra {

// Basic sphere class
class Sphere : virtual public Object, virtual public Renderable {
protected:
	float		_radius;
public:
	static constexpr char object_type[] = "Sphere";

	// Default constructor
	Sphere() = default;

	// Copy constructor
	Sphere(const Sphere &sphere)
			: Object(object_type, Transform()),
			Renderable(sphere.material().copy()) {}

	// Constructors
	Sphere(float radius)
			: Object(object_type, Transform()),
			_radius(radius) {}

	Sphere(const Sphere &sphere, const Transform &transform)
			: Object(object_type, transform),
			Renderable(sphere.material().copy()),
			_radius(sphere._radius) {}

	// Getters
	glm::vec3 center() const override {
		return _transform.position;
	}

	float radius() const {
		return _radius;
	}

	// Ray intersection
	float intersect(const Ray &ray) const override {
		glm::vec3 oc = ray.origin - _transform.position;
		float a = glm::dot(ray.direction, ray.direction);
		float b = 2.0f * glm::dot(oc, ray.direction);
		float c = glm::dot(oc, oc) - _radius * _radius;
		float discriminant = b * b - 4.0f * a * c;
		if (discriminant < 0.0f)
			return -1.0f;
		float t = (-b - std::sqrt(discriminant)) / (2.0f * a);
		if (t < 0.0f)
			t = (-b + std::sqrt(discriminant)) / (2.0f * a);
		return t;
	}

	// Virtual methods
	void save(std::ofstream &file) const override {
		file << "[SPHERE]\n";
		file << "radius=" << _radius << "\n";
		_material.save(file);
	}

	// Read sphere object from file
	static std::optional <Sphere> from_file
			(vk::raii::PhysicalDevice &phdev,
			vk::raii::Device &device,
			vk::raii::CommandPool &command_pool,
			std::ifstream &file,
			const std::string &scene_file) {
		std::string line;

		// Read radius
		float radius;
		std::getline(file, line);
		std::sscanf(line.c_str(), "radius=%f", &radius);

		// Read material header
		std::getline(file, line);
		if (line != "[MATERIAL]") {
			KOBRA_LOG_FUNC(error) << "Invalid kobra scene file format"
				<< " (missing [MATERIAL] header)\n";
			return std::nullopt;
		}

		// Read material
		bool success;
		auto mat = Material::from_file(
			phdev, device,
			command_pool,
			file, scene_file,
			success
		);

		if (!success) {
			KOBRA_LOG_FUNC(error) << "Invalid kobra scene file format (bad material)\n";
			return std::nullopt;
		}

		// Construct and return sphere
		Sphere sphere(radius);
		sphere.set_material(std::move(mat));

		return sphere;
	}
};

};

#endif

