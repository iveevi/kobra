////////////////////
// Texture module //
////////////////////

// Layout bindings for textures
layout (set = 0, binding = 10) buffer Textures
{
	// Start at the corresponding index,
	// each pixel is one vec4 as float data
	vec4 data[];
} textures;

// Layout bindings for texture infos
layout (set = 0, binding = 11) buffer TextureInfos
{
	// Format: -, width, height, channels
	uvec4 data[];
} texture_infos;

// Index of ith texture
uint texture_index(uint i)
{
	return texture_infos.data[i].x;
}

// Sample texture given spherical direction
vec4 sample_texture(vec3 dir, uint tid)
{
	// Get texture info
	uvec4 info = texture_infos.data[tid];
	uint width = info.y;
	uint height = info.z;
	uint channels = info.w;

	// Get texture coordinates
	vec2 uv = dir.xy;
	uv = uv * 0.5 + 0.5;

	// Get pixel index
	uint x = uint(uv.x * float(width));
	uint y = uint(uv.y * float(height));

	// Get pixel data
	uint index = y * width + x + texture_index(tid);
	vec4 pixel = textures.data[index];

	// Return pixel data
	return pixel;
}