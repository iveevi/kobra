#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Unix directory headers
#include <sys/types.h>
#include <dirent.h>

// Logger
#include "include/logger.hpp"
#include "include/init.hpp"
#include "include/ui/text.hpp"

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

struct File {
	// Icon object
	ui::Text text;
	std::string name;
};

void proc_res(DIR *res)
{
	dirent *dp;
	while ((dp = readdir(res))) {
		std::string file = dp->d_name;

		size_t len = file.length();
		if (file.length() < 4)
			continue;

		std::string ext = file.substr(len - 4);
		Logger::warn() << "\t\text = " << ext << "\n";

		if (ext == ".obj")
			Logger::ok("Object file.");
		else if (ext == ".mtl")
			Logger::ok("Material file.");
	}
}

int main()
{
	// Logger::start();
	init(false);

	Logger::ok("Starting file browser.");

	// Process the directory
	DIR *pdir = opendir(project.c_str());

	if (!pdir)
		Logger::error() << "Path \"" << project << "\" does not exist.\n";

	Logger::warn() << "pdir = " << pdir << "\n";

	dirent *dp;
	while ((dp = readdir(pdir))) {
		std::string file = dp->d_name;

		Logger::warn() << "\t" << file << "\n";

		if (file == "res") {
			Logger::warn() << "\t\tFOUND RES\n";

			std::string rdir = project + "/res";
			DIR *res = opendir(rdir.c_str());
			proc_res(res);
			closedir(res);
		}

		if (file.length() > 4 && file.substr(file.length() - 3) == ".hg")
			Logger::ok("Mercury configuration file");
	}
	closedir(pdir);
}

void mouse_callback(GLFWwindow*, double, double) {}
