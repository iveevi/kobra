#ifndef COMMON_H_
#define COMMON_H_

// Standard headers
#include <fstream>
#include <iostream>
#include <stdarg.h>
#include <string>
#include <vector>

// GLM header
#include <glm/glm.hpp>

// Engine headers
#include "logger.hpp"

namespace kobra {

namespace common {

// Check if a file exists
inline bool file_exists(const std::string &file)
{
	std::ifstream f(file);
	return f.good();
}

// Read file into a string
inline std::string read_file(const std::string &file)
{
	std::ifstream f(file);
	std::stringstream s;
	s << f.rdbuf();
	return s.str();
}

// Read file glob
inline std::vector <unsigned int> read_glob(const std::string &path)
{
	std::ifstream file(path, std::ios::ate | std::ios::binary);

	// Check that the file exists
	KOBRA_ASSERT(file.is_open(), "Failed to open file: " + path);

	// Get the file size
	size_t fsize = file.tellg();
	file.seekg(0);

	// Allocate memory for the file
	std::vector <unsigned int> buffer(fsize/sizeof(unsigned int));

	// Read the file
	file.read((char *) buffer.data(), fsize);
	file.close();

	return buffer;
}

// Get directory
inline std::string get_directory(const std::string &file)
{
	// Unix
#ifdef __unix__

	return file.substr(0, file.find_last_of('/'));

#endif

	// Windows

#ifdef _WIN32

	return file.substr(0, file.find_last_of('\\'));

#endif
}

// Relative or absolute path
inline std::string get_path(const std::string &file, const std::string &dir)
{
	std::string full = dir + "/" + file;
	if (file_exists(full))
		return full;
	return file;
}

// Printf to string
inline std::string sprintf(const char *fmt, ...)
{
	char buf[1024];
	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);
	return std::string(buf);
}

}

////////////////////
// Math functions //
////////////////////

// Closest distance between line segment and point
inline float distance(const glm::vec2 &a, const glm::vec2 &b, const glm::vec2 &p)
{
	glm::vec2 ab = b - a;
	glm::vec2 ap = p - a;
	float t = glm::dot(ap, ab) / glm::dot(ab, ab);
	if (t < 0.0f)
		return glm::length(p - a);
	if (t > 1.0f)
		return glm::length(p - b);
	return glm::length(p - a - t * ab);
}

// Project point onto plane, assuming origin is at (0, 0, 0)
inline glm::vec3 point_onto_plane(const glm::vec3 &point, const glm::vec3 &normal)
{
	return point - glm::dot(point, normal) * normal;
}

///////////////////////
// Simple structures //
///////////////////////

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

// Closest point between ray and point
inline glm::vec3 closest_point(const Ray &ray, const glm::vec3 &point)
{
	glm::vec3 ab = ray.direction;
	glm::vec3 ap = point - ray.origin;
	float t = glm::dot(ap, ab) / glm::dot(ab, ab);
	return ray.origin + t * ab;
}

}

#endif
