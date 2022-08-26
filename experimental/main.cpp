#include <thread>

#define KOBRA_PROFILING
#include "../include/profiler.hpp"

int main()
{
	// Task 1
	{
		KOBRA_PROFILE_TASK(Task 1);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	// Task 2
	{
		KOBRA_PROFILE_AUTO();
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		
		// Subtask 1
		{
			KOBRA_PROFILE_TASK(Subtask 1);
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
	}

	KOBRA_PROFILE_PRINT();
}
