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

// Methods
MaterialDaemon *make_material_daemon();

void subscribe(MaterialDaemon *, MaterialDaemon::Forward);
void signal_update(MaterialDaemon *, int32_t);
void update(MaterialDaemon *);

int32_t load(MaterialDaemon *, const std::filesystem::path &);
int32_t load(MaterialDaemon *, const Material &);

}

}
