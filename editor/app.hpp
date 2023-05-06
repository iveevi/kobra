#pragma once

// Standard headers
#include <queue>
#include <string>
#include <vector>

// Engine headers
#include "include/backend.hpp"

// Communication packet
struct Packet {
        std::string header;
        // TODO: variant of data types
        std::vector <int> data;
};

// Global communications sentinel
struct Application {
	float speed = 10.0f;
        kobra::Context context;
        std::queue <Packet> packets;
        std::string project;
};

extern Application g_application;
