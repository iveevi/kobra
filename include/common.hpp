#ifndef COMMON_H_
#define COMMON_H_

// Standard headers
#include <fstream>
#include <string>

// GLM header
#include <glm/glm.hpp>

namespace kobra {

namespace common {

// Check if a file exists
inline bool file_exists(const std::string &file)
{
	std::ifstream f(file);
	return f.good();
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

///////////////////////
// Simple structures //
///////////////////////

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

}

#endif
