#ifndef COMMON_H_
#define COMMON_H_

// Standard headers
#include <fstream>
#include <string>

namespace kobra {

namespace common {

// Check if a file exists
inline bool file_exists(const std::string &file)
{
	std::ifstream f(file);
	return f.good();
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

}

#endif
