// Include stb symbols in this module only
#define STB_IMAGE_IMPLEMENTATION

#include "include/model.hpp"
#include "include/logger.hpp"

// Standard headers
#include <iostream>

namespace mercury {

// Texture functions
Texture::Texture() {}

Texture::Texture(const std::string &file)
{
	// Allocate the texture
	glGenTextures(1, &id);

	// Retrieve data and dimensions
	int width;
	int height;
	int channels;

	unsigned char *data = stbi_load(
		file.c_str(), &width, &height,
		&channels, 0
	);

	if (!data) {
		Logger::error() << "Texture: Failed to load at path \""
			<< path << "\"" << std::endl;
		stbi_image_free(data);
	}

	// Check the image type
	GLenum format;

	if (channels == 1)
		format = GL_RED;
	else if (channels == 3)
		format = GL_RGB;
	else if (channels == 4)
		format = GL_RGBA;

	// Rest of the texture loading code
	glBindTexture(GL_TEXTURE_2D, id);
	glTexImage2D(GL_TEXTURE_2D, 0, format, width,
		height, 0, format, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
			GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	stbi_image_free(data);
}

Texture::Texture(const std::string &txt_path,
		const std::string &txt_dir,
		const std::string &txt_type)
		: type(txt_type), path(txt_path)
{
	std::string file = txt_dir + '/' + txt_path;

	// TODO: remove this duplicate sutff... (delegate constructors)

	// Allocate the texture
	glGenTextures(1, &id);

	// Retrieve data and dimensions
	int width;
	int height;
	int channels;

	unsigned char *data = stbi_load(
		file.c_str(), &width, &height,
		&channels, 0
	);

	if (!data) {
		Logger::error() << "Texture: Failed to load at path \""
			<< path << "\"" << std::endl;
		stbi_image_free(data);
	}

	// Check the image type
	GLenum format;

	if (channels == 1)
		format = GL_RED;
	else if (channels == 3)
		format = GL_RGB;
	else if (channels == 4)
		format = GL_RGBA;

	// Rest of the texture loading code
	glBindTexture(GL_TEXTURE_2D, id);
	glTexImage2D(GL_TEXTURE_2D, 0, format, width,
		height, 0, format, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
			GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	stbi_image_free(data);
}

// Mesh functions
Mesh::Mesh() {}

Mesh::Mesh(const AVertex &vertices,
		const ATexture &textures,
		const AIndices &indices)
		: _vertices(vertices),
		_textures(textures),
		_indices(indices)
{
	_init();
}

// TODO: clean
void Mesh::_init()
{
	glGenVertexArrays(1, &_vao);
	glGenBuffers(1, &_vbo);
	glGenBuffers(1, &_ebo);

	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);

	glBufferData(
		GL_ARRAY_BUFFER,
		_vertices.size() * sizeof(Vertex),
		&_vertices[0],
		GL_STATIC_DRAW
	);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, _indices.size() * sizeof(unsigned int),
			&_indices[0], GL_STATIC_DRAW);

	// vertex positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	// vertex normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
	// vertex texture coords
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texcoord));

	glBindVertexArray(0);
}

// TODO: clean
void Mesh::draw(Shader &shader)
{
	// Use the shader first
	shader.use();

	// TODO: check for wireframe (and/or vertex-dot) mode

	// bind appropriate textures
	unsigned int diffuseNr  = 1;
	unsigned int specularNr = 1;
	unsigned int normalNr   = 1;
	unsigned int heightNr   = 1;
	for (unsigned int i = 0; i < _textures.size(); i++)
	{
		glActiveTexture(GL_TEXTURE0 + i); // active proper texture unit before binding
		// retrieve texture number (the N in diffuse_textureN)
		std::string number;
		std::string name = _textures[i].type;
		if(name == "texture_diffuse")
			number = std::to_string(diffuseNr++);
		else if(name == "texture_specular")
			number = std::to_string(specularNr++); // transfer unsigned int to stream
		else if(name == "texture_normal")
			number = std::to_string(normalNr++); // transfer unsigned int to stream
		else if(name == "texture_height")
			number = std::to_string(heightNr++); // transfer unsigned int to stream

		// now set the sampler to the correct texture unit
		// TODO: use the shader method instead
		glUniform1i(glGetUniformLocation(shader.id, (name + number).c_str()), i);
		// and finally bind the texture
		glBindTexture(GL_TEXTURE_2D, _textures[i].id);
	}

	// draw mesh
	glBindVertexArray(_vao);
	glDrawElements(GL_TRIANGLES, _indices.size(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	// always good practice to set everything back to defaults once configured.
	glActiveTexture(GL_TEXTURE0);
}

// Model functions
Model::Model(const char *path)
{
	_load(path);
}

// Private methods
void Model::_load(const std::string &path)
{
	Assimp::Importer import;

	const aiScene *scene = import.ReadFile(path,
		aiProcess_Triangulate | aiProcess_FlipUVs);

	if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)
			|| !scene->mRootNode)  {
		Logger::error() << "Assimp error: "
			<< import.GetErrorString() << std::endl;
		return;
	}
	_dir = path.substr(0, path.find_last_of('/'));

	_proc_node(scene, scene->mRootNode);
}

void Model::_proc_node(const aiScene *scene, aiNode *node)
{
	// Process all meshes
	for (unsigned int i = 0; i < node->mNumMeshes; i++) {
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
		_meshes.push_back(_proc_mesh(scene, mesh));
	}

	// Recurse through each child node
	for (unsigned int i = 0; i < node->mNumChildren; i++)
		_proc_node(scene, node->mChildren[i]);
}

Mesh Model::_proc_mesh(const aiScene *scene, aiMesh *mesh)
{
	Mesh::AVertex vertices;
	Mesh::ATexture textures;
	Mesh::AIndices indices;

	// TODO: surely this can be made more efficient
	// TODO: separate into more functions
	for (unsigned int i = 0; i < mesh->mNumVertices; i++)
	{
		Vertex vertex;

		// Placeholder for temporary data
		glm::vec3 vector;

		// Set position
		vector.x = mesh->mVertices[i].x;
		vector.y = mesh->mVertices[i].y;
		vector.z = mesh->mVertices[i].z;
		vertex.position = vector;

		// Set normals
		if (mesh->HasNormals()) {
			vector.x = mesh->mNormals[i].x;
			vector.y = mesh->mNormals[i].y;
			vector.z = mesh->mNormals[i].z;
			vertex.normal = vector;
		}

		// Set texture coordinates
		if(mesh->mTextureCoords[0]) {
			glm::vec2 vec;
			vec.x = mesh->mTextureCoords[0][i].x;
			vec.y = mesh->mTextureCoords[0][i].y;
			vertex.texcoord = vec;
		} else {
			vertex.texcoord = glm::vec2(0.0f, 0.0f);
		}

		vertices.push_back(vertex);
	}

	// Process Mesh faces and corresponding indicies
	for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; j++)
			indices.push_back(face.mIndices[j]);
	}

	// Process materials
	// TODO: new function
	aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

	// Load all the textures
	ATexture diffuses = _load_textures(material, aiTextureType_DIFFUSE, "texture_diffuse");
	ATexture speculars = _load_textures(material, aiTextureType_SPECULAR, "texture_specular");
	ATexture normals = _load_textures(material, aiTextureType_HEIGHT, "texture_normal");
	ATexture heights = _load_textures(material, aiTextureType_AMBIENT, "texture_height");

	// Transfer all teh textures
	textures.insert(textures.end(), diffuses.begin(), diffuses.end());
	textures.insert(textures.end(), speculars.begin(), speculars.end());
	textures.insert(textures.end(), normals.begin(), normals.end());
	textures.insert(textures.end(), heights.begin(), heights.end());

	// Construct and return Mesh object
	return Mesh(vertices, textures, indices);
}

Model::ATexture Model::_load_textures(
		aiMaterial *mat,
		aiTextureType type,
		std::string name)
{
	ATexture out;
	for (size_t i = 0; i < mat->GetTextureCount(type); i++) {
		aiString str;
		mat->GetTexture(type, i, &str);

		bool cached = false;
		for(unsigned int j = 0; j < _cached_textures.size(); j++) {
			if(std::strcmp(_cached_textures[j].path.data(), str.C_Str()) == 0)
			{
				out.push_back(_cached_textures[j]);
				cached = true;
				break;
			}
		}

		// Load the texture if it is not cached
		if(!cached) {
			/* Texture texture;
			texture.id = TextureFromFile(str.C_Str(), _dir);
			texture.type = type;
			texture.path = str.C_Str(); */
			Texture texture(str.C_Str(), _dir, name);

			out.push_back(texture);
			_cached_textures.push_back(texture); // add to loaded textures
		}
	}

	return out;
}

// Public methods
void Model::draw(Shader &shader)
{
	for (Mesh &mesh : _meshes)
		mesh.draw(shader);
}

}
