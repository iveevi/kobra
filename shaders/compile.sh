echo "Compiling shaders..."

# Create binary directories
mkdir -p bin bin/ui bin/generic bin/raster

# Compile GENERIC mode shaders
# glslc -fshader-stage=compute rt/normal.glsl -o bin/generic/normal.spv
# glslc -fshader-stage=compute rt/heatmap.glsl -o bin/generic/heatmap.spv
glslc -fshader-stage=compute rt/progressive_path_tracer.glsl -o bin/generic/progressive_path_tracer.spv

glslc -fshader-stage=vertex rt/postproc/postproc.vert -o bin/generic/postproc_vert.spv
glslc -fshader-stage=fragment rt/postproc/postproc.frag -o bin/generic/postproc_frag.spv

# Compile GUI mode shaders
glslc -fshader-stage=vertex ui/basic.vert -o bin/ui/basic_vert.spv
glslc -fshader-stage=fragment ui/basic.frag -o bin/ui/basic_frag.spv

glslc -fshader-stage=vertex ui/sprite.vert -o bin/ui/sprite_vert.spv
glslc -fshader-stage=fragment ui/sprite.frag -o bin/ui/sprite_frag.spv

glslc -fshader-stage=vertex ui/glyph.vert -o bin/ui/glyph_vert.spv
glslc -fshader-stage=fragment ui/glyph.frag -o bin/ui/glyph_frag.spv

glslc -fshader-stage=fragment ui/bitmap.frag -o bin/ui/bitmap_frag.spv

# Compile rasteization shaders
glslc -fshader-stage=vertex raster/vertex.vert -o bin/raster/vertex.spv
glslc -fshader-stage=fragment raster/color.frag -o bin/raster/color_frag.spv
glslc -fshader-stage=fragment raster/plain_color.frag -o bin/raster/plain_color_frag.spv
glslc -fshader-stage=fragment raster/normal.frag -o bin/raster/normal_frag.spv
glslc -fshader-stage=fragment raster/blinn_phong.frag -o bin/raster/blinn_phong_frag.spv
