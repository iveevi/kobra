#ifndef KOBRA_OPTIX_SBT_H_
#define KOBRA_OPTIX_SBT_H_

// Engine headers
#include "../bbox.hpp"
#include "../cuda/material.cuh"
#include "../vertex.hpp"

namespace kobra {

namespace optix {

// Hit data record
struct Hit {	
	// Transform data
	glm::mat4 model;

	// Mesh data
	// glm::vec2		*texcoords;
	glm::uvec3		*triangles;
	
	/* glm::vec3		*vertices;

	glm::vec3 		*normals;
	glm::vec3		*tangents;
	glm::vec3		*bitangents; */

	Vertex			*vertices;

	// Auto UV mapping parameters
	BoundingBox		bbox;

	// Material and textures
	cuda::Material		material;

	struct {
		cudaTextureObject_t	diffuse;
		cudaTextureObject_t	emission;
		cudaTextureObject_t	normal;
		cudaTextureObject_t	roughness;
		cudaTextureObject_t	specular;

		bool			has_diffuse = false;
		bool			has_emission = false;
		bool			has_normal = false;
		bool			has_roughness = false;
		bool			has_specular = false;
	} textures;
};


}

}

#endif
