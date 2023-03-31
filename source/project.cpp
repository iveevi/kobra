#include "../include/project.hpp"
#include <sstream>

namespace kobra {

std::string trim_whitespace(const std::string &str)
{
        std::string result = str;
        result.erase(0, result.find_first_not_of(' '));
        result.erase(result.find_last_not_of(' ') + 1);
        return result;
}

Material load_material(std::ifstream &file)
{
        Material mat;

        int name_length;
        file.read((char *) &name_length, sizeof(int));
        std::string name;
        name.resize(name_length);
        file.read((char *) name.data(), name_length);

        glm::vec3 diffuse;
        file.read((char *) &diffuse, sizeof(glm::vec3));

        glm::vec3 specular;
        file.read((char *) &specular, sizeof(glm::vec3));

        glm::vec3 ambient;
        file.read((char *) &ambient, sizeof(glm::vec3));

        glm::vec3 emission;
        file.read((char *) &emission, sizeof(glm::vec3));

        float roughness;
        file.read((char *) &roughness, sizeof(float));

        float refraction;
        file.read((char *) &refraction, sizeof(float));

        Shading type;
        file.read((char *) &type, sizeof(Shading));

        int normal_texture_length;
        file.read((char *) &normal_texture_length, sizeof(int));
        std::string normal_texture;
        normal_texture.resize(normal_texture_length);
        file.read((char *) normal_texture.data(), normal_texture_length);

        int diffuse_texture_length;
        file.read((char *) &diffuse_texture_length, sizeof(int));
        std::string diffuse_texture;
        diffuse_texture.resize(diffuse_texture_length);
        file.read((char *) diffuse_texture.data(), diffuse_texture_length);

        int specular_texture_length;
        file.read((char *) &specular_texture_length, sizeof(int));
        std::string specular_texture;
        specular_texture.resize(specular_texture_length);
        file.read((char *) specular_texture.data(), specular_texture_length);

        int emission_texture_length;
        file.read((char *) &emission_texture_length, sizeof(int));
        std::string emission_texture;
        emission_texture.resize(emission_texture_length);
        file.read((char *) emission_texture.data(), emission_texture_length);

        int roughness_texture_length;
        file.read((char *) &roughness_texture_length, sizeof(int));
        std::string roughness_texture;
        roughness_texture.resize(roughness_texture_length);
        file.read((char *) roughness_texture.data(), roughness_texture_length);

        mat.name = name;
        mat.diffuse = diffuse;
        mat.specular = specular;
        mat.ambient = ambient;
        mat.emission = emission;
        mat.roughness = roughness;
        mat.refraction = refraction;
        mat.type = type;
        mat.normal_texture = normal_texture;
        mat.albedo_texture = diffuse_texture;
        mat.specular_texture = specular_texture;
        mat.emission_texture = emission_texture;
        mat.roughness_texture = roughness_texture;

        return mat;
}

Submesh load_mesh(std::ifstream &file)
{
        int num_vertices;
        file.read((char *) &num_vertices, sizeof(int));

        int num_indices;
        file.read((char *) &num_indices, sizeof(int));

        std::vector <Vertex> vertices;
        vertices.resize(num_vertices);
        file.read((char *) vertices.data(), sizeof(Vertex) * num_vertices);

        std::vector <uint32_t> indices;
        indices.resize(num_indices);
        file.read((char *) indices.data(), sizeof(int) * num_indices);

        return Submesh { vertices, indices, 0};
}

void s_load_scene(const std::filesystem::path &path, const Context &context, Scene &scene, std::ifstream &file)
{
        std::vector <std::string> lines;

        std::string line;
        while (std::getline(file, line)) {
                printf("Line: %s\n", line.c_str());
                lines.push_back(line);
        }

        struct Element {
                enum Type {
                        ENTITY,
                        MATERIALS
                } type;

                using FieldValue = std::variant <
                        int,
                        float,
                        Transform,
                        std::string,
                        std::vector <int>,
                        std::vector <std::string>
                >;

                std::map <std::string, FieldValue> fields;
        };

        std::vector <Element> elements;
        Element *current_element = nullptr;

        int i = 0;
        while (i < lines.size()) {
                const std::string &line = lines[i++];

                // Skip empty lines
                if (line.empty())
                        continue;

                // Skip comments
                if (line[0] == '#')
                        continue;

                // Check if it's a new element
                if (line[0] == '@') {
                        printf("New element: %s\n", line.c_str());
                        std::istringstream iss(line.substr(1));

                        std::string type;
                        iss >> type;

                        if (type == "entity") {
                                elements.push_back(Element { Element::ENTITY });
                                printf("Entity\n");

                                std::string name = line.substr(7);
                                elements.back().fields["name"] = trim_whitespace(name);
                        } else if (type == "materials") {
                                elements.push_back(Element { Element::MATERIALS });
                                printf("Materials\n");
                        } else {
                                KOBRA_LOG_FILE(Log::WARN) << "Unknown element type: " << type << std::endl;
                                continue;
                        }
                        
                        current_element = &elements.back();
                }

                // Check if it's a field
                if (line[0] == '.') {
                        if (current_element == nullptr) {
                                KOBRA_LOG_FILE(Log::WARN) << "Field without element: " << line << std::endl;
                                continue;
                        }

                        std::string field = line.substr(1);
                        printf("Field: %s\n", field.c_str());

                        if (field.rfind("transform", 0) == 0) {
                                printf("\tTransform\n");

                                float vec[9];
                                std::istringstream iss(field.substr(10));
                                for (int i = 0; i < 9; i++)
                                        iss >> vec[i];

                                printf("\t\t%f %f %f\n", vec[0], vec[1], vec[2]);
                                printf("\t\t%f %f %f\n", vec[3], vec[4], vec[5]);
                                printf("\t\t%f %f %f\n", vec[6], vec[7], vec[8]);

                                Transform transform {
                                        { vec[0], vec[1], vec[2] },
                                        { vec[3], vec[4], vec[5] },
                                        { vec[6], vec[7], vec[8] }
                                };

                                current_element->fields["transform"] = transform;
                        } else if (field.rfind("mesh", 0) == 0) {
                                printf("\tMesh\n");

                                int mesh_count = std::stoi(field.substr(5));

                                std::vector <std::string> meshes;
                                std::vector <int> materials;

                                for (int j = 0; j < mesh_count; j++) {
                                        std::istringstream iss(lines[i++]);

                                        std::string mesh;
                                        int material_index;

                                        // Skip whitespace
                                        iss >> std::ws;
                                        
                                        // Cut the trailing comma
                                        std::getline(iss, mesh, ',');
                                        iss >> material_index;

                                        printf("\t\t%s, %d\n", mesh.c_str(), material_index);
                                        meshes.push_back(mesh);
                                        materials.push_back(material_index);
                                }

                                current_element->fields["mesh"] = meshes;
                                current_element->fields["material"] = materials;
                        } else if (field.rfind("renderable", 0) == 0) {
                                printf("\tRenderable\n");
                                current_element->fields["renderable"] = true;
                        } else if (field.rfind("list", 0) == 0) {
                                if (current_element->type != Element::MATERIALS) {
                                        KOBRA_LOG_FILE(Log::WARN) << "List field without materials element: " << line << std::endl;
                                        continue;
                                }
                                
                                if (field.length() <= 5) {
                                        KOBRA_LOG_FILE(Log::WARN) << "List field without count: " << line << std::endl;
                                        continue;
                                }

                                std::string count = field.substr(5);
                                elements.back().fields["count"] = std::stoi(count);

                                printf("\tList of materials, count: %s\n", count.c_str());

                                std::vector <std::string> materials;
                                for (int j = 0; j < std::stoi(count); j++) {
                                        std::istringstream iss(lines[i++]);

                                        std::string material;
                                        iss >> material;

                                        printf("\t\t%s\n", material.c_str());
                                        materials.push_back(material);
                                }

                                current_element->fields["list"] = materials;
                        } else if (field.rfind("camera", 0) == 0) {
                                float fov;
                                float aspect;

                                std::istringstream iss(field.substr(7));
                                iss >> fov >> aspect;

                                printf("\tCamera, fov: %f, aspect: %f\n", fov, aspect);

                                current_element->fields["fov"] = fov;
                                current_element->fields["aspect"] = aspect;
                        } else {
                                KOBRA_LOG_FILE(Log::WARN) << "Unknown field: " << field << std::endl;
                                continue;
                        }
                }
        }

        // Print the elements
        printf("-----------------------------------------\n");
        printf("Elements: %lu\n", elements.size());
        for (Element &element : elements) {
                if (element.type == Element::ENTITY) {
                        printf("Entity:\n");
                } else if (element.type == Element::MATERIALS) {
                        printf("Materials:\n");
                }

                for (auto &field : element.fields) {
                        printf("\t%s: ", field.first.c_str());

                        if (std::holds_alternative <int> (field.second)) {
                                printf("%d\n", std::get <int> (field.second));
                        } else if (std::holds_alternative <float> (field.second)) {
                                printf("%f\n", std::get <float> (field.second));
                        } else if (std::holds_alternative <Transform> (field.second)) {
                                Transform &transform = std::get <Transform> (field.second);

                                glm::vec3 position = transform.position;
                                printf("Transform: %f %f %f\n", position.x, position.y, position.z);
                        } else if (std::holds_alternative <std::string> (field.second)) {
                                printf("%s\n", std::get <std::string> (field.second).c_str());
                        } else if (std::holds_alternative <std::vector <std::string>> (field.second)) {
                                printf("\n");
                                for (const std::string &str : std::get <std::vector <std::string>> (field.second))
                                        printf("\t\t%s\n", str.c_str());
                        } else {
                                printf("Unknown type\n");
                        }
                }
        }

        // Initilize the ECS
        scene.ecs = std::make_shared <ECS> ();

        // Add the entities
        bool loaded_materials = false;
        for (Element &element : elements) {
                if (element.type == Element::MATERIALS) {
                        // First clear the current global list
                        // TODO: should also signal the daemon
                        Material::all.clear();
                        if (element.fields.find("list") == element.fields.end()) {
                                KOBRA_LOG_FILE(Log::WARN) << "No materials listed, missing .list?" << std::endl;
                                continue;
                        }

                        std::vector <std::string> &materials = std::get
                                <std::vector <std::string>> (element.fields["list"]);

                        for (const std::string &material : materials) {
                                printf("Material: %s\n", material.c_str());

                                // Check the assets path
                                std::filesystem::path material_path = path / "assets" / material;
                                if (!std::filesystem::exists(material_path)) {
                                        KOBRA_LOG_FILE(Log::WARN) << "Material file does not exist: " << material_path << std::endl;
                                        continue;
                                }

                                std::ifstream material_file(material_path);

                                Material mat = load_material(material_file);
                                Material::all.push_back(mat);
                        }

                        loaded_materials = true;
                        continue;
                }

                // Make sure all the materials are loaded
                if (!loaded_materials) {
                        KOBRA_LOG_FILE(Log::WARN) << "Entities defined before materials" << std::endl;
                        continue;
                }

                // Create the entity
                std::string name = std::get <std::string> (element.fields["name"]);
                Entity entity = scene.ecs->make_entity(name);

                // Add the components
                for (auto &field : element.fields) {
                        if (field.first == "transform") {
                                Transform &transform = std::get <Transform> (field.second);
                                entity.get <Transform> () = transform;
                        } else if (field.first == "mesh") {
                                std::vector <std::string> &meshes = std::get
                                        <std::vector <std::string>> (field.second);
                                
                                std::vector <int> &materials = std::get
                                        <std::vector <int>> (element.fields["material"]);

                                std::vector <Submesh> mesh_list;
                                for (int i = 0; i < meshes.size(); i++) {
                                        std::string &mesh = meshes[i];
                                        int material = materials[i];

                                        if (material >= Material::all.size()) {
                                                KOBRA_LOG_FILE(Log::WARN) << "Material index out of range: " << material << std::endl;
                                                continue;
                                        }

                                        printf("Mesh: %s, material: %d\n", mesh.c_str(), material);

                                        // Check the assets path
                                        std::filesystem::path mesh_path = path / ".cache" / mesh;
                                        if (!std::filesystem::exists(mesh_path)) {
                                                KOBRA_LOG_FILE(Log::WARN) << "Mesh file does not exist: " << mesh_path << std::endl;
                                                continue;
                                        }

                                        std::ifstream mesh_file(mesh_path);

                                        Submesh mesh_object = load_mesh(mesh_file);
                                        mesh_object.material_index = material;
                                        mesh_list.push_back(mesh_object);                                        
                                }

                                if (mesh_list.empty()) {
                                        KOBRA_LOG_FILE(Log::WARN) << "No meshes loaded for entity: " << name << std::endl;
                                        continue;
                                }

                                entity.add <Mesh> (mesh_list);
                        } else if (field.first == "renderable") {
                                // TODO: no need for mesh component, only renderable should be enough
                                // cache still contains all necessary meshes...
                                // scene.ecs->add_component <Renderable> (entity);

                                // Make sure the entity has a mesh by now
                                if (!entity.exists <Mesh> ()) {
                                        KOBRA_LOG_FILE(Log::WARN) << "Entity does not have a mesh: " << name << std::endl;
                                        continue;
                                }

                                Mesh *mesh = &entity.get <Mesh> ();
                                entity.add <Renderable> (context, mesh);
                        } else if (field.first == "fov") {
                                // Also get aspect
                                float fov = std::get <float> (field.second);
                                float aspect = std::get <float> (element.fields["aspect"]);

                                printf("Camera: fov = %f, aspect = %f\n", fov, aspect);

                                entity.add <Camera> (fov, aspect);
                        }
                }
        }
}

Scene &Project::load_scene(const Context &context, int index)
{
        // If negative, use the default scene
        if (index == -1)
                index = default_scene_index;
        
        // Make sure there are at least some scenes
        if (index >= scenes.size())
                throw std::runtime_error("No scenes loaded");

        // Skip if the scene is already loaded
        if (scenes[index].ecs)
                return scenes[index];

        // Check if the scene exists
        if (index >= scenes_files.size())
                throw std::runtime_error("Scene does not exist");

        // Load the scene into the scene list
        std::string scene_file = scenes_files[index];

        std::filesystem::path path = directory;
        path /= scene_file;

        printf("Loading scene from path: %s, stem = %s\n", path.c_str(), path.stem().c_str());

        // scenes[index].load(context, path.string());
        std::ifstream file(path);
        s_load_scene(directory, context, scenes[index], file);
        scenes[index].name = path.stem();

        return scenes[index];
}

}