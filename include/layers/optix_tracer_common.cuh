#ifndef KOBRA_LAYERS_OPTIX_TRACER_COMMON_H_
#define KOBRA_LAYERS_OPTIX_TRACER_COMMON_H_

// OptiX headers
#include <optix.h>

namespace kobra {

struct Params
{
	uchar4			*image;
	unsigned int		image_width;
	unsigned int		image_height;
	float3			cam_eye;
	float3			cam_u, cam_v, cam_w;
	OptixTraversableHandle	handle;
};

struct RayGenData
{
	// No data needed
};

struct MissData
{
	// Background color
	// TODO: background texture
	float3			bg_color;

	cudaTextureObject_t	bg_tex;
};

struct HitGroupData
{
	// Materials
	struct Material {
		float3		diffuse;
		float3		specular;
		float3		emission;
		float3		ambient;
		float		shininess;
		float		roughness;
		float		refraction;
	};

	Material		material;

	float2			*texcoords;
	float3			*vertices;
	uint3			*triangles;

	float3			*normals;
	float3			*tangents;
	float3			*bitangents;

	struct {
		cudaTextureObject_t	diffuse;
		cudaTextureObject_t	normal;

		bool			has_diffuse;
		bool			has_normal;
	} textures;
};

}

#endif
