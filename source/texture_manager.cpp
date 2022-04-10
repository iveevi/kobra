#include "../include/texture_manager.hpp"

namespace kobra {

/////////////////////////////
// Static member variables //
/////////////////////////////

TextureManager::TextureCache TextureManager::_cached;
std::mutex TextureManager::_mutex;

////////////////////
// Static methods //
////////////////////

const Texture &TextureManager::load(const std::string &path, int channels)
{
	_mutex.lock();
	if (_cached.find(path) != _cached.end()) {
		_mutex.unlock();
		return _cached[path];
	}
	_mutex.unlock();

	Texture texture = load_image_texture(path, channels);

	_mutex.lock();
	_cached[path] = texture;
	const Texture &t = _cached[path];
	_mutex.unlock();

	return t;
}

}
