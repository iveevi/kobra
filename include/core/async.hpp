#ifndef KOBRA_CORE_ASYNC_H_
#define KOBRA_CORE_ASYNC_H_

// Standard headers
#include <mutex>
#include <queue>
#include <thread>

namespace kobra {

namespace core {

struct AsyncTask {
        std::mutex mutex;
        std::thread *thread;
};

}

}

#endif