mkdir -p bin
glslc --target-env=vulkan1.3 shaders/raytrace.rchit -o bin/raytrace.rchit.spv
glslc --target-env=vulkan1.3 shaders/raytrace.rgen -o bin/raytrace.rgen.spv
glslc --target-env=vulkan1.3 shaders/raytrace.rmiss -o bin/raytrace.rmiss.spv
