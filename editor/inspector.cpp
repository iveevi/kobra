#include "inspector.hpp"

Inspector *make_inspector(kobra::System *system)
{
        Inspector *inspector = new Inspector;
        inspector->system = system;
        return inspector;
}

void select(Inspector *inspector, int32_t entity)
{
        inspector->selected = entity;
}

void render(Inspector *inspector)
{
        // Empty window if nothing is selected
        if (inspector->selected == -1) {
                ImGui::Begin("Inspector");
                ImGui::End();
                return;
        }

        // Load the entity
        kobra::Entity &entity = inspector->system->get_entity(inspector->selected);

        ImGui::Begin("Inspector");

        ImGui::Text("Name: %s", entity.name.c_str());

        // Transform
        ImGui::Text("Transform");

        kobra::Transform &transform = entity.get <kobra::Transform> ();

        {
                ImGui::DragFloat3("Position", &transform.position.x, 0.1f);
                ImGui::SameLine();
                if (ImGui::Button("Reset##position")) {
                        std::cout << "Resetting position" << std::endl;
                        transform.position = glm::vec3 { 0.0f };
                }
        }

        {
                ImGui::DragFloat3("Rotation", &transform.rotation.x, 0.1f);
                ImGui::SameLine();
                if (ImGui::Button("Reset##rotation")) {
                        std::cout << "Resetting rotation" << std::endl;
                        transform.rotation = glm::vec3 { 0.0f };
                }
        }

        {
                ImGui::DragFloat3("Scale", &transform.scale.x, 0.1f);
                ImGui::SameLine();
                if (ImGui::Button("Reset##scale")) {
                        std::cout << "Resetting scale" << std::endl;
                        transform.scale = glm::vec3 { 1.0f };
                }
        }

        // Mesh
        // TODO: meshes themself should not be componants...
        if (entity.exists <kobra::Mesh> ()) {
                ImGui::Text("Mesh");
        }
        
        // Renderable
        if (entity.exists <kobra::Renderable> ()) {
                if (ImGui::CollapsingHeader("Renderable")) {
                        ImGui::Text("Number of submeshes: %lu", entity.get <kobra::Mesh> ().submeshes.size());
                }
        }

        // Camera
        if (entity.exists <kobra::Camera> ()) {
                ImGui::Text("Camera");
        }

        ImGui::End();
}
