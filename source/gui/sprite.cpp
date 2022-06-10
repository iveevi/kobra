// Engine headers
#include "../../include/gui/sprite.hpp"
#include "../../include/gui/layer.hpp"

namespace kobra {

namespace gui {
	
// Latch onto a layer
void Sprite::latch(LatchingPacket &lp)
{
	_ds = lp.layer->serve_sprite_ds();
	if (*_sampler)
		bind_ds(_device, _ds, _sampler, _image_data, 0);
}

}

}
