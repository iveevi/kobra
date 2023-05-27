// Standard headers
#include <sstream>

// Engine headers
#include "include/project.hpp"

namespace kobra {

// Saving projects
static std::string transcribe_submesh(const Submesh &submesh) {
        // TODO: generate the tangent and bitangent vectors
        // if they are not present yet

        // Setup the stream
        std::ostringstream stream;

        int vertices = submesh.vertices.size();
        int indices = submesh.indices.size();

        stream.write((char *) &vertices, sizeof(int));
        stream.write((char *) &indices, sizeof(int));
        stream.write((char *) submesh.vertices.data(), sizeof(Vertex) * vertices);
        stream.write((char *) submesh.indices.data(), sizeof(int) * indices);

        return stream.str();
}

// Transcribe material into binary data
constexpr char material_fmt[] = R"(name: %s
diffuse: %f,%f,%f
specular: %f,%f,%f
emission: %f,%f,%f
roughness: %f
refraction: %f
type: %d
albedo_texture: %s
normal_texture: %s
roughness_texture: %s)";

static std::string transcribe_material(const Material &material) {
        char buffer[1 << 12] = {0};

        sprintf(buffer, material_fmt,
                material.name.c_str(),
                material.diffuse.x, material.diffuse.y, material.diffuse.z,
                material.specular.x, material.specular.y, material.specular.z,
                material.emission.x, material.emission.y, material.emission.z,
                material.roughness,
                material.refraction,
                material.type,
                material.diffuse_texture.c_str(),
                material.normal_texture.c_str(),
                material.roughness_texture.c_str());

        return buffer;
}

// Save project
void Project::save()
{
        // TODO: detect parts that have changed...
        printf("Saving to %s\n", directory.c_str());
        std::filesystem::path path = directory;

        // Create the necessary directories
        std::filesystem::create_directory(path);			// Root directory
        std::filesystem::create_directory(path / ".cache");	// Cache directory
        std::filesystem::create_directory(path / "assets");	// Assets directory (user home)

        std::filesystem::path cache_path = path / ".cache";
        std::filesystem::path assets_path = path / "assets";

        // Collect all SUBMESHES to populate the cache
        // TODO: need to find similar enough meshes (e.g. translated or
        // scaled)
        std::set <const Submesh *> submesh_cache;
        for (auto &scene : scenes)
                scene.populate_mesh_cache(submesh_cache);

        // ID each submesh in the cache
        std::vector <std::pair <const Submesh *, std::string>> submesh_ids;
        std::map <const Submesh *, size_t> submesh_id_map;

        // TODO: use the name of the entity for the mesh...
        size_t id = 0;
        for (auto &submesh : submesh_cache) {
                submesh_ids.push_back({submesh, "submesh-" + std::to_string(id)});
                submesh_id_map[submesh] = id;
                id++;
        }

        tf::Taskflow taskflow;
        tf::Executor executor;

        taskflow.for_each(submesh_ids.begin(), submesh_ids.end(),
                [&](const auto &pr) {
                        // TODO: ID each submesh in the cache...
                        std::filesystem::path filename = cache_path/(pr.second + ".submesh");
                        std::ofstream file(filename, std::ios::binary);

                        // Write the submesh to the file
                        std::string data = transcribe_submesh(*pr.first);
                        file.write(data.data(), data.size());

                        file.close();
                }
        );

        executor.run(taskflow).wait();

        // Collect all materials
        // TODO: save in the same location as creation (in the assets
        // directory...)

        // int index = 0;
        // for (auto &mat : Material::all) {
        //         mat.name = "material-" + std::to_string(index++);
        // }

        // For now, store in the cache directory
        // taskflow.for_each(Material::all.begin(), Material::all.end(),
        //         [&](const Material &mat) {
        //                 // TODO: ID each submesh in the cache...
        //                 std::filesystem::path filename = assets_path/(mat.name + ".mat");
        //                 std::ofstream file(filename, std::ios::binary);
        //
        //                 // Write the submesh to the file
        //                 std::string data = transcribe_material(mat);
        //                 file.write(data.data(), data.size());
        //
        //                 file.close();
        //         }
        // );
       
        const auto &materials = material_daemon->materials;
        taskflow.for_each(materials.begin(), materials.end(),
                [&](const Material &mat) {
                        // TODO: ID each submesh in the cache...
                        std::filesystem::path filename = assets_path/(mat.name + ".mat");
                        std::ofstream file(filename, std::ios::binary);

                        // Write the submesh to the file
                        std::string data = transcribe_material(mat);
                        file.write(data.data(), data.size());

                        file.close();
                }
        );
        
        executor.run(taskflow).wait();

        // Scene description file (.kobra)
        for (auto &scene : scenes) {
                std::filesystem::path filename = path / (scene.name + ".kobra");
                std::ofstream file(filename);

                // Write the scene description
                for (auto &entity : *scene.system) {
                        file << "\n@entity " << entity.name << "\n";

                        const Transform &transform = entity.get <Transform> ();
                        file << ".transform "
                                << transform.position.x << " " << transform.position.y << " " << transform.position.z << " "
                                << transform.rotation.x << " " << transform.rotation.y << " " << transform.rotation.z << " "
                                << transform.scale.x << " " << transform.scale.y << " " << transform.scale.z << "\n";

                        if (entity.exists <Mesh> ()) {
                                file << ".mesh " << entity.get <Mesh> ().submeshes.size() << "\n";
                                auto &submeshes = entity.get <Mesh> ().submeshes;
                                for (auto &submesh : submeshes) {
                                        size_t id = submesh_id_map[&submesh];
                                        // TODO: find the path instead...
                                        // std::string material = Material::all[submesh.material_index].name;
                                        std::string material = materials[submesh.material_index].name;
                                        file << "\tsubmesh-" << id << ".submesh, " << material << ".mat\n";
                                }

                                // TODO: material indices...
                        }

                        if (entity.exists <Renderable> ()) {
                                file << ".renderable\n";
                                // TODO: material ids
                        }

                        if (entity.exists <Camera> ()) {
                                const Camera &camera = entity.get <Camera> ();
                                file << ".camera "
                                        << camera.fov << " "
                                        << camera.aspect << "\n";
                        }
                }

                file.close();
        }

        // Top level project file, describing all the scenes (.den)
        std::filesystem::path filename = path / "project.den";
        std::ofstream file(filename);

        // Write all the scenes
        file << "@scenes " << scenes.size() << "\n";
        for (auto &scene : scenes)
                file << scene.name << ".kobra\n";

        // TODO: use fmt library

        // TODO: other project information

        file.close();
}

// Loading projects
static std::string trim_whitespace(const std::string &str)
{
        std::string result = str;
        result.erase(0, result.find_first_not_of(' '));
        result.erase(result.find_last_not_of(' ') + 1);
        return result;
}

static Submesh load_mesh(std::ifstream &file)
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

static void s_load_scene(const std::filesystem::path &path, const Context &context, Scene &scene, MaterialDaemon *material_daemon, std::ifstream &file)
{
        std::vector <std::string> lines;

        std::string line;
        while (std::getline(file, line)) {
                printf("Line: %s\n", line.c_str());
                lines.push_back(line);
        }

        struct Element {
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
                                // elements.push_back(Element { Element::ENTITY });
                                elements.push_back(Element {});
                                printf("Entity\n");

                                std::string name = line.substr(7);
                                elements.back().fields["name"] = trim_whitespace(name);
                        } else if (type == "materials") {
                                // elements.push_back(Element { Element::MATERIALS });
                                elements.push_back(Element {});
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
                                std::vector <std::string> materials_paths;

                                for (int j = 0; j < mesh_count; j++) {
                                        std::istringstream iss(lines[i++]);

                                        std::string mesh;
                                        std::string material_path;

                                        // Skip whitespace
                                        iss >> std::ws;
                                        
                                        // Cut the trailing comma
                                        std::getline(iss, mesh, ',');
                                        iss >> material_path;

                                        printf("\t\t%s, %s\n", mesh.c_str(), material_path.c_str());
                                        meshes.push_back(mesh);
                                        materials_paths.push_back(material_path);
                                }

                                current_element->fields["mesh"] = meshes;
                                current_element->fields["material"] = materials_paths;
                        } else if (field.rfind("renderable", 0) == 0) {
                                printf("\tRenderable\n");
                                current_element->fields["renderable"] = true;
                        } else if (field.rfind("list", 0) == 0) {
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
        // printf("-----------------------------------------\n");
        // printf("Elements: %lu\n", elements.size());
        // for (Element &element : elements) {
        //         if (element.type == Element::ENTITY) {
        //                 printf("Entity:\n");
        //         } else if (element.type == Element::MATERIALS) {
        //                 printf("Materials:\n");
        //         }
        //
        //         for (auto &field : element.fields) {
        //                 printf("\t%s: ", field.first.c_str());
        //
        //                 if (std::holds_alternative <int> (field.second)) {
        //                         printf("%d\n", std::get <int> (field.second));
        //                 } else if (std::holds_alternative <float> (field.second)) {
        //                         printf("%f\n", std::get <float> (field.second));
        //                 } else if (std::holds_alternative <Transform> (field.second)) {
        //                         Transform &transform = std::get <Transform> (field.second);
        //
        //                         glm::vec3 position = transform.position;
        //                         printf("Transform: %f %f %f\n", position.x, position.y, position.z);
        //                 } else if (std::holds_alternative <std::string> (field.second)) {
        //                         printf("%s\n", std::get <std::string> (field.second).c_str());
        //                 } else if (std::holds_alternative <std::vector <std::string>> (field.second)) {
        //                         printf("\n");
        //                         for (const std::string &str : std::get <std::vector <std::string>> (field.second))
        //                                 printf("\t\t%s\n", str.c_str());
        //                 } else {
        //                         printf("Unknown type\n");
        //                 }
        //         }
        // }

        // Initilize the system
        scene.system = std::make_shared <System> (material_daemon);

        // Add the entities
        for (Element &element : elements) {
                // Create the entity
                std::string name = std::get <std::string> (element.fields["name"]);
                Entity entity = scene.system->make_entity(name);

                // Add the components
                for (auto &field : element.fields) {
                        if (field.first == "transform") {
                                Transform &transform = std::get <Transform> (field.second);
                                entity.get <Transform> () = transform;
                        } else if (field.first == "mesh") {
                                std::vector <std::string> &meshes = std::get
                                        <std::vector <std::string>> (field.second);
                                
                                std::vector <std::string> &material_paths = std::get
                                        <std::vector <std::string>> (element.fields["material"]);

                                std::vector <Submesh> mesh_list;
                                for (int i = 0; i < meshes.size(); i++) {
                                        std::string &mesh = meshes[i];
                                        std::string &material_path = material_paths[i];

                                        std::filesystem::path material_path_full = path / "assets" / material_path;
                                        // if (!std::filesystem::exists(material_path_full)) {
                                        //         KOBRA_LOG_FILE(Log::WARN) << "Material file does not exist: " << material_path_full << std::endl;
                                        //         continue;
                                        // }
                                        //
                                        // std::ifstream material_file(material_path_full);
                                        // Material material = load_material(material_file);
                                        // int32_t index = Material::all.size();
                                        // Material::all.push_back(material);
                                        int32_t index = load(material_daemon, material_path_full);

                                        printf("Mesh: %s, material: %s\n", mesh.c_str(), material_path_full.c_str());

                                        // Check the assets path
                                        std::filesystem::path mesh_path = path / ".cache" / mesh;
                                        if (!std::filesystem::exists(mesh_path)) {
                                                KOBRA_LOG_FILE(Log::WARN) << "Mesh file does not exist: " << mesh_path << std::endl;
                                                continue;
                                        }

                                        std::ifstream mesh_file(mesh_path);

                                        Submesh mesh_object = load_mesh(mesh_file);
                                        mesh_object.material_index = index;
                                        mesh_list.push_back(mesh_object);                                        

                                        // Material::all = material_daemon->materials;
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

        // Make sure there is a valid material daemon
        if (!material_daemon)
                throw std::runtime_error("No material daemon, is a project loaded?");

        // Skip if the scene is already loaded
        if (scenes[index].system)
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
        s_load_scene(directory, context, scenes[index], material_daemon, file);
        scenes[index].name = path.stem();

        return scenes[index];
}

}
