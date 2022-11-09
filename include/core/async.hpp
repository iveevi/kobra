#ifndef KOBRA_CORE_ASYNC_H_
#define KOBRA_CORE_ASYNC_H_

// Standard headers
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace kobra {

namespace core {

struct AsyncTask {
        std::thread *thread = nullptr;

        enum Status {
                eRunning,
                eFinished
        } status = eRunning;

        // Launch an asynchronous task
        AsyncTask(std::function <void ()> task) {
                auto wrapper = [&](AsyncTask *at) {
                        task();
                        at->status = eFinished;
                };

                thread = new std::thread(wrapper, this);
        }

        // Destructor
        ~AsyncTask() {
                if (thread) {
                        thread->join();
                        delete thread;
                }
        }

        // Wait for the task to finish
        void wait() {
                if (thread) {
                        thread->join();
                        delete thread;
                        thread = nullptr;
                }
        }
};

}

}

#endif