#pragma once

// ImGui headers
#include <imgui.h>

// Engine headers
#include "include/ecs.hpp"

struct Inspector {
        kobra::ECS *ecs = nullptr;
        int32_t selected = -1;
};

Inspector *make_inspector(kobra::ECS *);
void select(Inspector *, int32_t);
void render(Inspector *);
