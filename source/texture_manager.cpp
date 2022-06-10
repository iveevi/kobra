#include "../include/texture_manager.hpp"

namespace kobra {

/////////////////////////////
// Static member variables //
/////////////////////////////

TextureManager::TextureMap TextureManager::_texture_map;
TextureManager::TextureData TextureManager::_texture_data;
std::mutex TextureManager::_mutex;

////////////////////
// Static methods //
////////////////////

const ImageData &TextureManager::load(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &dev,
		const vk::raii::CommandPool &command_pool,
		const std::string &path,
		int channels)
{
	_mutex.lock();
	if (_texture_map.find(path) != _texture_map.end()) {
		_mutex.unlock();
		size_t index = _texture_map[path];
		return _texture_data[index];
	}
	_mutex.unlock();

	// TODO: convert channels to image format
	ImageData img = make_image(phdev, dev,
		command_pool, path,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	_mutex.lock();
	_texture_data.emplace_back(std::move(img));
	_texture_map[path] = _texture_data.size() - 1;
	const ImageData &ret = _texture_data.back();
	_mutex.unlock();

	return ret;
}

}
