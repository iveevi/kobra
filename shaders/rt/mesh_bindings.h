#ifndef KOBRA_RT_MESH_BINDINGS_H_
#define KOBRA_RT_MESH_BINDINGS_H_

// Essential buffers
const int MESH_BINDING_PIXELS		= 0;
const int MESH_BINDING_VIEWPORT		= 1;
const int MESH_BINDING_VERTICES		= 2;
const int MESH_BINDING_TRIANGLES	= 3;
const int MESH_BINDING_TRANSFORMS	= 4;
const int MESH_BINDING_BVH		= 5;
const int MESH_BINDING_MATERIALS	= 6;

// Buffers for lights
const int MESH_BINDING_LIGHTS		= 7;
const int MESH_BINDING_LIGHT_INDICES	= 8;

// Samplers
const int MESH_BINDING_ALBEDO		= 9;
const int MESH_BINDING_NORMAL_MAPS	= 10;

// TODO: mesh roughness/bump map

#endif
