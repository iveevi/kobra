#ifndef KOBRA_TEXTURE_MANAGER_H_
#define KOBRA_TEXTURE_MANAGER_H_

// Standard headers
#include <mutex>
#include <string>
#include <unordered_map>

// Engine headers
#include "texture.hpp"

namespace kobra {

// Caches all loaded textures , globally
// TODO: later also allow multihreaded loading
class TextureManager {
	using TextureCache = std::unordered_map <std::string, Texture>;
	static TextureCache	_cached;
	static std::mutex	_mutex;
public:
	// Loads a texture from file and caches it
	static const Texture &load(const std::string &, int = -1);
};

}

#endif
