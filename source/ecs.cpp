#include "include/ecs.hpp"

namespace mercury {

// ECS specializations
template <>
Object *ECS::get <Object> (size_t i)
{
	return objects[i];
}

}
