#ifndef SHADER_H_
#define SHADER_H_

namespace mercury {

class Shader {
public:
	Shader(const char *, const char *);

	void use();

	// Variables
	unsigned int id;
};

}

#endif
