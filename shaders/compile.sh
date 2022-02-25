# Create binary directories
mkdir -p bin bin/generic bin/generic

# Compile GENERIC mode shaders
glslc -fshader-stage=compute pixel.glsl -o bin/generic/pixel.spv
glslc -fshader-stage=vertex postproc/postproc.vert -o bin/generic/pp_vert.spv
glslc -fshader-stage=fragment postproc/postproc.frag -o bin/generic/pp_frag.spv