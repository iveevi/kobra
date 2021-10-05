#ifndef SHADER_H_
#define SHADER_H_

// Standard headers
#include <string>

// GLM
#include <glm/glm.hpp>

namespace mercury {

class Shader {
	unsigned int	_vertex;
	unsigned int	_fragment;

#ifdef MERCURY_DEBUG

	int		_id;

	static int	_current;

	static int get_nid() {
		static int count = 0;

		return (++count);
	}

#endif

public:
	Shader();
	Shader(const char *, const char *);

	// TODO: destructor?

	// Methods
	void use() const;
	void set_vertex_shader(const char *);
	void set_fragment_shader(const char *);
	void compile();

	// Variables
	unsigned int id;

	// Setters
	void set_bool(const std::string &, bool) const;
	void set_int(const std::string &, int) const;
	void set_float(const std::string &, float) const;

	void set_vec2(const std::string &, const glm::vec2 &) const;
	void set_vec2(const std::string &, float, float) const;

	void set_vec3(const std::string &, const glm::vec3 &) const;
	void set_vec3(const std::string &, float, float, float) const;

	void set_vec4(const std::string &, const glm::vec4 &) const;
	void set_vec4(const std::string &, float, float, float, float) const;

	void set_mat2(const std::string &, const glm::mat2 &) const;
	void set_mat3(const std::string &, const glm::mat3 &) const;
	void set_mat4(const std::string &, const glm::mat4 &) const;

	// Static methods
	static Shader from_source(const char *, const char *);
};

}

#endif
