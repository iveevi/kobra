#ifndef KOBRA_TEXTURE_MANAGER_H_
#define KOBRA_TEXTURE_MANAGER_H_

// Standard headers
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

// Engine headers
// #include "texture.hpp"
#include "backend.hpp"

namespace kobra {

// Caches all loaded textures , globally
// TODO: later also allow multihreaded loading
class TextureManager {
	// TODO: One cache per device?
	using TextureMap = std::map <std::string, size_t>;
	using TextureData = std::vector <ImageData>;

	static TextureMap	_texture_map;
	static TextureData	_texture_data;
	static std::mutex	_mutex;
public:
	// Loads a texture from file and caches it
	static const ImageData &load(const vk::raii::PhysicalDevice &,
			const vk::raii::Device &,
			const vk::raii::CommandPool &,
			const std::string &,
			int = -1);
};

}

#endif
