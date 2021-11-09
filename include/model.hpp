#ifndef MODEL_H_
#define MODEL_H_

// Standard headers
#include <string>
#include <vector>

// GLFW headers
#include "../glad/glad.h"
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/glm.hpp>

// Assimp
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// STB headers
#include "../thirdparty/stb/stb_image.h"

// Engine headers
#include "shader.hpp"

namespace mercury {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texcoord;
};

// TODO: put texture into a different set of files
struct Texture {
	unsigned int id;
	std::string type;
	std::string path;

	Texture();
	Texture(const std::string &);

	// TODO: whats the last parameter for?
	Texture(const std::string &, const std::string &, const std::string &);
};

class Mesh {
public:
	// Public aliases
	using AVertex = std::vector <Vertex>;
	using ATexture = std::vector <Texture>;
	using AIndices = std::vector <unsigned int>;
private:
	AVertex 	_vertices;
	ATexture	_textures;
	AIndices	_indices;

	unsigned int			_vao;
	unsigned int			_vbo;
	unsigned int			_ebo;

	void _init();
public:
	Mesh(const AVertex &, const ATexture &, const AIndices &);

	void draw(Shader &);
};

class Model {
public:
	// Public aliases
	using ATexture = std::vector <Texture>;
	using AMesh = std::vector <Mesh>;
private:
	ATexture	_cached_textures;
	AMesh		_meshes;
	std::string	_dir;
	// bool		_gamma;

	void _load(const std::string &);

	void _proc_node(const aiScene *, aiNode *);
        Mesh _proc_mesh(const aiScene *, aiMesh *);

        ATexture _load_textures(aiMaterial *, aiTextureType, std::string);
public:
	Model(const char *);

	void draw(Shader &);
};

}

#endif
