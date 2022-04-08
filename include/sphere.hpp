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

	// Constructor
	Sphere(float radius)
			: Object(object_type, Transform()),
			_radius(radius) {}

	Sphere(const Sphere &sphere, const Transform &transform)
			: Object(object_type, transform),
			Renderable(sphere.material()),
			_radius(sphere._radius) {}

	// Getters
	const glm::vec3 &center() const {
		return _transform.position;
	}

	float radius() const {
		return _radius;
	}

	// Virtual methods
	void save(std::ofstream &file) const override {
		file << "[SPHERE]\n";
		file << "radius=" << _radius << "\n";
		_material.save(file);
	}

	// Read sphere object from file
	static std::optional <Sphere> from_file(const Vulkan::Context &ctx,
			const VkCommandPool &command_pool,
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
			KOBRA_LOG_FUNC(error) << "Invalid kobra scene file format\n";
			return std::nullopt;
		}

		// Read material
		std::optional <Material> material = Material::from_file(ctx,
			command_pool, file, scene_file
		);

		if (!material) {
			KOBRA_LOG_FUNC(error) << "Invalid kobra scene file format\n";
			return std::nullopt;
		}

		// Construct and return sphere
		Sphere sphere(radius);
		sphere.set_material(material.value());
		return sphere;
	}
};

};

#endif

