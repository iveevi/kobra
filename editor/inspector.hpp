#pragma once

// ImGui headers
#include <imgui.h>

// Engine headers
#include "include/system.hpp"

struct Inspector {
        kobra::System *system= nullptr;
        int32_t selected = -1;
};

Inspector *make_inspector(kobra::System *);
void select(Inspector *, int32_t);
void render(Inspector *);
