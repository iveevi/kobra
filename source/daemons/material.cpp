#include "include/daemons/material.hpp"

namespace kobra {

MaterialDaemon *make_material_daemon()
{
        return new MaterialDaemon;
}

void subscribe(MaterialDaemon *daemon, MaterialDaemon::Forward forward)
{
        daemon->forward.push_back(forward);
}

void signal_update(MaterialDaemon *daemon, int32_t id)
{
        daemon->status[id] = 1;
}

void update(MaterialDaemon *daemon)
{
        std::set <int32_t> updated;
        for (int i = 0; i < daemon->materials.size(); i++) {
                if (daemon->status[i] == 1) {
                        updated.insert(i);
                        daemon->status[i] = 0;
                }
        }

        for (auto &forward : daemon->forward) {
                for (int32_t id : updated)
                        forward->push(id);
        }
}

static Material load_material(std::ifstream &file)
{
        Material material;

        char buf_name[1024] = {0};
	char buf_albedo[1024] = {0};
	char buf_normal[1024] = {0};
	char buf_roughness[1024] = {0};
        int32_t type = 0;

        std::string name_line;
        std::getline(file, name_line);
        std::sscanf(name_line.c_str(), "name: %s", buf_name);

        std::string diffuse_line;
        std::getline(file, diffuse_line);
        std::sscanf(diffuse_line.c_str(), "diffuse: %f,%f,%f", &material.diffuse.x, &material.diffuse.y, &material.diffuse.z);

        std::string specular_line;
        std::getline(file, specular_line);
        std::sscanf(specular_line.c_str(), "specular: %f,%f,%f", &material.specular.x, &material.specular.y, &material.specular.z);

        std::string ambient_line;
        std::getline(file, ambient_line);
        std::sscanf(ambient_line.c_str(), "ambient: %f,%f,%f", &material.ambient.x, &material.ambient.y, &material.ambient.z);

        std::string emission_line;
        std::getline(file, emission_line);
        std::sscanf(emission_line.c_str(), "emission: %f,%f,%f", &material.emission.x, &material.emission.y, &material.emission.z);

        std::string roughness_line;
        std::getline(file, roughness_line);
        std::sscanf(roughness_line.c_str(), "roughness: %f", &material.roughness);

        std::string refraction_line;
        std::getline(file, refraction_line);
        std::sscanf(refraction_line.c_str(), "refraction: %f", &material.refraction);

        std::string type_line;
        std::getline(file, type_line);
        std::sscanf(type_line.c_str(), "type: %d", &type);

        std::string albedo_line;
        std::getline(file, albedo_line);
        std::sscanf(albedo_line.c_str(), "albedo_texture: %s", buf_albedo);

        std::string normal_line;
        std::getline(file, normal_line);
        std::sscanf(normal_line.c_str(), "normal_texture: %s", buf_normal);

        std::string roughness_texture_line;
        std::getline(file, roughness_texture_line);
        std::sscanf(roughness_texture_line.c_str(), "roughness_texture: %s", buf_roughness);

        material.name = buf_name;
        material.albedo_texture = buf_albedo;
        material.normal_texture = buf_normal;
        material.roughness_texture = buf_roughness;
        material.type = (Shading) type;

        return material;
}

int32_t load(MaterialDaemon *daemon, const std::filesystem::path &path)
{
        // TODO: lookup for paths as well?
        std::ifstream file(path);
        if (!file.is_open())
                return -1;

        Material material = load_material(file);
        if (daemon->lookup.find(material.name) != daemon->lookup.end())
                return daemon->lookup[material.name];

        int32_t id = daemon->materials.size();
        daemon->materials.push_back(material);
        daemon->lookup[material.name] = id;
        daemon->status.push_back(0);

        return id;
}

int32_t load(MaterialDaemon *daemon, const Material &material)
{
        if (daemon->lookup.find(material.name) != daemon->lookup.end())
                return daemon->lookup[material.name];

        int32_t id = daemon->materials.size();
        daemon->materials.push_back(material);
        daemon->lookup[material.name] = id;
        daemon->status.push_back(0);

        return id;
}

}
