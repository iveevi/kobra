#include "../include/object.hpp"

namespace kobra {

//////////////////////
// Static variables //
//////////////////////

int Object::_name_id = 0;

////////////////////
// Static methods //
////////////////////

std::string Object::_generate_name()
{
	return "Object " + std::to_string(_name_id++);
}

}
