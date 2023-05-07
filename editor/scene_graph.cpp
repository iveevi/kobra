#include "scene_graph.hpp"

using namespace kobra;
	
void SceneGraph::set_scene(const Scene *scene)
{
        m_scene = scene;
}

void SceneGraph::render()
{
        ImGui::Begin("Scene Graph");
        
        if (m_scene != nullptr) {
                auto &ecs = *m_scene->ecs;

                // Button transparent background
                for (auto &entity : ecs) {
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
                                        box.submeshes[0].material_index = Material::all.size();
                                        Material::all.push_back(Material::default_material());
                                        auto &entity = m_scene->ecs->make_entity("Box");
                                        entity.add <Mesh> (box);
                                        entity.add <Renderable> (g_application.context, &entity.get <Mesh> ());
                                }

                                if (ImGui::MenuItem("Plane")) {
                                        Mesh plane = Mesh::plane();
                                        plane.submeshes[0].material_index = Material::all.size();
                                        Material::all.push_back(Material::default_material());
                                        auto &entity = m_scene->ecs->make_entity("Plane");
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
