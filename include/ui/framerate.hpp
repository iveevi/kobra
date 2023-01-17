#ifndef KOBRA_UI_FRAMERATE_H_
#define KOBRA_UI_FRAMERATE_H_

// Standard headers
#include <vector>
#include <algorithm>
#include <functional>

#include <iostream>

// ImGUI headers
#include <imgui/imgui.h>

// ImPlot headers
#include <implot/implot.h>
#include <implot/implot_internal.h>

// Engine headers
#include "attachment.hpp"
#include "../timer.hpp"

namespace kobra {

namespace ui {

class FramerateAttachment : public ImGuiAttachment {
public:
	FramerateAttachment() {
		m_timer.start();
		m_framerates.reserve(LIMIT);
		m_times.reserve(LIMIT);
		m_getter = []() {
			return ImGui::GetIO().Framerate;
		};
	}

	FramerateAttachment(std::function <float ()> getter) {
		m_timer.start();
		m_framerates.reserve(LIMIT);
		m_getter = getter;
	}

	void render() override {
		float fps = m_getter();

		m_framerates.push_back(fps);
		m_times.push_back(m_timer.elapsed_start()/1e6);
		
		// TODO: constant time range instead
		static constexpr float TIME_RANGE = 5.0f;

		while (m_times.back() - m_times.front() > TIME_RANGE) {
			m_framerates.erase(m_framerates.begin());
			m_times.erase(m_times.begin());
		}

		/* if (m_framerates.size() > LIMIT + 1)
			m_framerates.erase(m_framerates.begin());

		if (m_times.size() > LIMIT + 1)
			m_times.erase(m_times.begin()); */
		
		float max_fps = *std::max_element(m_framerates.begin(), m_framerates.end());
		max_fps = std::max(60.0f, max_fps);

		ImGui::Begin("Framerate");
		ImGui::Text("FPS: %d", (int) fps);
		
		float min_time = m_times.front();
		float max_time = m_times.back();

		if (ImPlot::BeginPlot("Framerate")) {
			ImPlot::SetupAxes("Time", "Framerate");
			ImPlot::SetupAxesLimits(
				min_time, max_time,
				0, max_fps + 5.0f,
				ImPlotCond_Always
			);

			ImPlot::PlotLine("FPS",
				m_times.data(),
				m_framerates.data(),
				m_framerates.size()
			);

			ImPlot::EndPlot();
		}

		ImGui::End();

		// TODO: graph (options)
	}
private:
	Timer m_timer;
	std::function <float ()> m_getter;
	std::vector <float> m_framerates;
	std::vector <float> m_times;
	
	static constexpr int LIMIT = 100;

};

}

}

#endif
