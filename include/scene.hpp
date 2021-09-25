#ifndef SCENE_H_
#define SCENE_H_

// Standard headers
#include <vector>

namespace mercury {

// TODO: create an include for this
class Camera;

class Scene {
	std::vector <Camera> _cams;
};

}

#endif
