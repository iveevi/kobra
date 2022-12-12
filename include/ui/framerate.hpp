#ifndef KOBRA_UI_FRAMERATE_H_
#define KOBRA_UI_FRAMERATE_H_

// Standard headers
#include <vector>
#include <algorithm>

// ImGUI headers
#include <imgui/imgui.h>

// ImPlot headers
#include <implot/implot.h>
#include <implot/implot_internal.h>

// Engine headers
#include "attachment.hpp"

namespace kobra {

namespace ui {

class FramerateAttachment : public ImGuiAttachment {
public:
	static constexpr int LIMIT = 100;

	FramerateAttachment() {
		m_history.reserve(LIMIT);
	}

	void render() override {
		float fps = ImGui::GetIO().Framerate;

		m_history.push_back(fps);
		if (m_history.size() > LIMIT + 1)
			m_history.erase(m_history.begin());
		
		float max_fps = *std::max_element(m_history.begin(), m_history.end());

		ImGui::Begin("Framerate");
		ImGui::Text("FPS: %d", (int) fps);
		
		ImPlot::SetNextAxesLimits(0, LIMIT, 0, 200);

		if (ImPlot::BeginPlot("Framerate")) {
			ImPlot::PlotLine("FPS", m_history.data(), m_history.size());
			ImPlot::EndPlot();
		}

		ImGui::End();

		// TODO: graph (options)
	}
private:
	std::vector <float> m_history;
};

}

}

#endif
