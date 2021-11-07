#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Unix directory headers
#include <sys/types.h>
#include <dirent.h>

// Logger
#include "include/logger.hpp"

using namespace std;
using namespace mercury;

const std::string project = "resources/project";

// TODO: there should be a .mercury file in each project directory
struct Project {
	std::string name;
	std::string path;	// Absolute path

	struct Config {
		bool anti_aliasing = false;
	} config;
};

void proc_res(DIR *res)
{
	dirent *dp;
	while ((dp = readdir(res))) {
		if (dp->d_namlen < 4)
			continue;

		const char *ext	= &dp->d_name[dp->d_namlen - 4];
		Logger::warn() << "\t\text = " << ext << "\n";

		if (strcmp(".obj", ext) == 0)
			Logger::ok("Object file.");
		else if (strcmp(".mtl", ext) == 0)
			Logger::ok("Material file.");
	}
}

int main()
{
	Logger::start();

	Logger::ok("Starting file browser.");

	// Process the directory
	DIR *pdir = opendir(project.c_str());

	if (!pdir)
		Logger::error() << "Path \"" << project << "\" does not exist.\n";

	Logger::warn() << "pdir = " << pdir << "\n";

	dirent *dp;
	while ((dp = readdir(pdir))) {
		Logger::warn() << "\t" << dp->d_name << "\n";

		if (strcmp("res", dp->d_name) == 0) {
			Logger::warn() << "\t\tFOUND RES\n";

			std::string rdir = project + "/res";
			DIR *res = opendir(rdir.c_str());
			proc_res(res);
			closedir(res);
		}
	}
	closedir(pdir);
}

void mouse_callback(GLFWwindow*, double, double) {}
