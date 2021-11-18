#ifndef SAVE_H_
#define SAVE_H_

// Engine headers
#include "include/archetype.hpp"

namespace mercury {

namespace filesystem {

// Saving meshes

// Save an Archetype context
void save(const Archetype &, const std::string &);

}

}

#endif