#include "material_editor.hpp"
	
void MaterialEditor::render()
{
        ImGui::Begin("Material editor");
        if (material_index < 0) {
                ImGui::Text("No material selected");
                ImGui::End();
                return;
        }

        // Transfer material properties to the material preview renderer
        mp->index = material_index;

        // TODO: resize to full size of the window
        ImVec2 size = ImGui::GetWindowSize();
        ImGui::Image(dset_material_preview, ImVec2(256, 256));

        ImVec2 pmin = ImGui::GetItemRectMin();
        ImVec2 pmax = ImGui::GetItemRectMax();

        input_context.material_preview.min = glm::vec2 { pmin.x, pmin.y };
        input_context.material_preview.max = glm::vec2 { pmax.x, pmax.y };

        // Check if it is a new material
        bool is_not_loaded = m_prev_material_index != material_index;
        m_prev_material_index = material_index;

        // For starters, print material data
        ImGui::Text("Material data:");
        ImGui::Separator();

        kobra::Material *material = &md->materials[material_index];

        glm::vec3 diffuse = material->diffuse;
        glm::vec3 specular = material->specular;
        float roughness = material->roughness;

        // Decompose the emission if it is not loaded
        if (is_not_loaded) {
                emission_base = glm::vec3(0.0f);
                emission_strength = 0.0f;

                // If any component is greater than 1, normalize it
                // TODO: use RGBE encoding...
                glm::vec3 emission = material->emission;
                if (emission.r > 1.0f || emission.g > 1.0f || emission.b > 1.0f) {
                        emission_strength = glm::length(emission);
                        emission_base = emission / emission_strength;
                }
        }

        bool updated_material = false;

        if (ImGui::ColorEdit3("Diffuse", &diffuse.r)) {
                material->diffuse = diffuse;
                updated_material = true;
        }

        if (ImGui::ColorEdit3("Specular", &specular.r)) {
                material->specular = specular;
                updated_material = true;
        }

        // TODO: remove ambient from material
        // TODO: use an HSL color picker + intensity slider
        if (ImGui::ColorEdit3("Emission", &emission_base.r)) {
                material->emission = emission_strength * emission_base;
                updated_material = true;
        }

        if (ImGui::SliderFloat("Intensity", &emission_strength, 0.0f, 1000.0f)) {
                material->emission = emission_strength * emission_base;
                updated_material = true;
        }

        // TODO: emission intensity
        if (ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f)) {
                material->roughness = std::max(roughness, 0.001f);
                updated_material = true;
        }

        // Transmission index of refraction
        if (ImGui::SliderFloat("IOR", &material->refraction, 1.0f, 3.0f))
                updated_material = true;

        // TODO: option for transmission
        bool transmission = (material->type == eTransmission);
        if (ImGui::Checkbox("Transmission", &transmission)) {
                material->type = transmission ? eTransmission : eDiffuse;
                updated_material = true;
        }

        ImGui::Separator();

        if (material->has_albedo()) {
                ImGui::Text("Diffuse Texture:");

                std::string diffuse_path = material->diffuse_texture;
                if (is_not_loaded)
                        m_diffuse_set = imgui_allocate_image(diffuse_path);

                ImGui::Image(m_diffuse_set, ImVec2(256, 256));
                ImGui::Separator();
        }

        if (material->has_normal()) {
                ImGui::Text("Normal Texture:");

                std::string normal_path = material->normal_texture;
                if (is_not_loaded)
                        m_normal_set = imgui_allocate_image(normal_path);

                ImGui::Image(m_normal_set, ImVec2(256, 256));
                ImGui::Separator();
        }

        // Notify the daemon that the material has been updated
        if (updated_material) {
                std::cout << "Updating material " << material_index << std::endl;
                signal_update(md, material_index);
        }

        // End
        ImGui::End();
}
