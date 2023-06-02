// Engine headers
#include "include/daemons/material.hpp"

// Editor headers
#include "scene_graph.hpp"

using namespace kobra;

int duplicate_count(System *system, const std::string &name)
{
        int max = 0;
        for (auto &entity : system->entities) {
                // Look for entities "[name] (count)"
                if (entity.name.find(name) == 0) {
                        std::string suffix = entity.name.substr(name.size());
                        if (suffix.size() > 0 && suffix[0] == ' ') {
                                suffix = suffix.substr(1);
                                if (std::all_of(suffix.begin(), suffix.end(), isdigit))
                                        max = std::max(max, std::stoi(suffix));
                        }
                }
        }

        return max;
}

void SceneGraph::set_scene(const Scene *scene)
{
        m_scene = scene;
}

void SceneGraph::render()
{
        System *system = m_scene->system.get();
        MaterialDaemon *md = system->material_daemon;

        ImGui::Begin("Scene Graph");
        
        if (m_scene != nullptr) {
                auto &system = *m_scene->system;

                // Button transparent background
                for (auto &entity : system) {
                        ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));

                        uint32_t width = ImGui::GetContentRegionAvail().x;
                        if (ImGui::Button(entity.name.c_str(), ImVec2(width, 0))) {
                                std::cout << "Selected entity: " << entity.name << std::endl;
                                g_application.packets.push(Packet {
                                        .header = "select_entity",
                                        .data = { entity.id }
                                });
                        }

                        ImGui::PopStyleVar();
                }
        }

        // Open a popup when the user right clicks on the scene graph
        if (ImGui::BeginPopupContextWindow()) {
                if (ImGui::BeginMenu("Add Entity")) {
                        if (ImGui::BeginMenu("Renderable")) {
                                if (ImGui::MenuItem("Box")) {
                                        Mesh box = Mesh::box();
                                        // TODO: method to request new material from a daemon...
                                        int count = duplicate_count(system, "Box");
                                        int32_t index = load(md, Material::default_material("box_" + std::to_string(count + 1) + "_material"));
                                        box.submeshes[0].material_index = index;

                                        auto &entity = system->make_entity("Box " + std::to_string(count + 1));
                                        entity.add <Mesh> (box);
                                        entity.add <Renderable> (g_application.context, &entity.get <Mesh> ());
                                }

                                if (ImGui::MenuItem("Plane")) {
                                        Mesh plane = Mesh::plane();
                                        
                                        int count = duplicate_count(system, "Plane");
                                        int32_t index = load(md, Material::default_material("plane_" + std::to_string(count + 1) + "_material"));
                                        plane.submeshes[0].material_index = index;

                                        auto &entity = system->make_entity("Plane" + std::to_string(count + 1));
                                        entity.add <Mesh> (plane);
                                        entity.add <Renderable> (g_application.context, &entity.get <Mesh> ());
                                }

                                ImGui::EndMenu();
                        }

                        ImGui::EndMenu();
                }

                ImGui::EndPopup();
        }

        ImGui::End();
}
