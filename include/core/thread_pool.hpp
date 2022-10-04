#ifndef KOBRA_CORE_THREAD_POOL_H_
#define KOBRA_CORE_THREAD_POOL_H_

// Standard headers
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

namespace kobra {

namespace core {

// Regular task queue system
using Task = std::function <void ()>;
using TaskQueue = std::queue <Task>;

inline void run_tasks
		(TaskQueue& tasks,
		 int pool_size = std::thread::hardware_concurrency())
{
	std::mutex mutex;
	std::vector <std::thread> threads;

	for (int i = 0; i < pool_size; ++i) {
		threads.emplace_back(
			[&]() {
				while (true) {
					mutex.lock();
					if (tasks.empty()) {
						mutex.unlock();
						break;
					}

					auto task = tasks.front();
					tasks.pop();

					mutex.unlock();

					task();
				}
			}
		);
	}

	for (auto& thread : threads)
		thread.join();
}

// Implicit task queue system
template <class Task>
using TaskExecutor = std::function <void (const Task &)>;

template <class Task>
void run_tasks
		(const TaskExecutor <Task> &executor,
		 std::queue <Task> tasks,
		 int pool_size = std::thread::hardware_concurrency())
{
	std::mutex mutex;
	std::vector <std::thread> threads;

	for (int i = 0; i < pool_size; ++i) {
		threads.emplace_back(
			[&]() {
				while (true) {
					mutex.lock();
					if (tasks.empty()) {
						mutex.unlock();
						break;
					}

					auto task = tasks.front();
					tasks.pop();

					mutex.unlock();

					executor(task);
				}
			}
		);
	}

	for (auto &thread : threads)
		thread.join();
}

// Implicitly generated
template <class Task>
using TaskGenerator = std::function <std::optional <Task> ()>;

template <class Task>
void run_tasks
		(const TaskExecutor <Task> &executor,
		 const TaskGenerator <Task> &generator,
		 int pool_size = std::thread::hardware_concurrency())
{
	std::mutex mutex;
	std::vector <std::thread> threads;

	for (int i = 0; i < pool_size; ++i) {
		threads.emplace_back(
			[&]() {
				while (true) {
					mutex.lock();

					auto task = generator();
					if (!task) {
						mutex.unlock();
						break;
					}

					mutex.unlock();

					executor(*task);
				}
			}
		);
	}

	for (auto &thread : threads)
		thread.join();
}

}

}

#endif
