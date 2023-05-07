#pragma once

// Standard headers
#include <filesystem>
#include <map>
#include <queue>
#include <vector>

// Engine headers
#include "../material.hpp"

namespace kobra {

namespace daemons {

struct MaterialDaemon {
        using Forward = std::queue <int32_t> *;

        std::map <std::string, int32_t> lookup;
        std::vector <Material> materials;
        std::vector <int32_t> status;
        std::vector <Forward> forward;
};

MaterialDaemon *make_material_daemon()
{
        return new MaterialDaemon;
}

// int32_t load_material(MaterialDaemon *daemon, const std::filesystem::path &path)
// {
// }

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

}

}
