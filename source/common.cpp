// Standard headers
#include <fstream>

// Engine headers
#include "include/common.hpp"

// Overload printing glm::vec3
std::ostream &operator<<(std::ostream &os, const glm::vec3 &vec)
{
	return os << "<" << vec.x << ", " << vec.y
		<< ", " << vec.z << ">";
}

// Miscellaneous
static void error_file(const char *path)
{
	mercury::Logger::error() << "Could not load file \""
	    << path << "\"\n";
	exit(-1);

	// TODO: should throw a retrievable exception later
	// a vector of potential files...?
}

std::string read_code(const char *path)
{
	std::ifstream file;
	std::string out;

	file.open(path);
	if (!file)
		error_file(path);

	std::stringstream ss;
	ss << file.rdbuf();
	out = ss.str();

	return out;
}

std::string read_code(const std::string &path)
{
	return read_code(path.c_str());
}
