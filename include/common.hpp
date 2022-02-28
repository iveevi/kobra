#ifndef COMMON_H_
#define COMMON_H_

// Standard headers
#include <fstream>
#include <string>

// Check if a file exists
inline bool file_exists(const std::string &file)
{
	std::ifstream f(file);
	return f.good();
}

#endif