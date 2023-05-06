#pragma once

// ImGui headers
#include <imgui.h>

// Engine headers
#include "include/material.hpp"
#include "include/mesh.hpp"
#include "include/scene.hpp"
#include "include/ui/attachment.hpp"

// Editor headers
#include "common.hpp"

struct SceneGraph : public kobra::ui::ImGuiAttachment {
	const kobra::Scene *m_scene = nullptr;
public:
	SceneGraph() = default;

	void set_scene(const kobra::Scene *scene);
	void render() override;
};
