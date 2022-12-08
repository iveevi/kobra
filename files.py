import os

# Definitions
kobra_sources = [
	'source/*',
	'source/io/*',
	'source/layers/*.cpp',
	'source/layers/basilisk.cu',
	'source/layers/denoiser.cu',
	'source/layers/wssr_grid.cu',
	'source/layers/optix_tracer.cu',
	# 'source/asmodeus/*',
    'source/optix/core.cu'
]

glslang_sources = [
    'thirdparty/glslang/SPIRV/GlslangToSpv.cpp',
    'thirdparty/glslang/StandAlone/ResourceLimits.cpp'
]

imgui_sources = [
    'thirdparty/imgui/imgui.cpp',
    'thirdparty/imgui/imgui_demo.cpp',
    'thirdparty/imgui/imgui_draw.cpp',
    'thirdparty/imgui/imgui_tables.cpp',
    'thirdparty/imgui/imgui_widgets.cpp',
    'thirdparty/imgui/backends/imgui_impl_glfw.cpp',
    'thirdparty/imgui/backends/imgui_impl_vulkan.cpp',
]

implot_sources = [
    'thirdparty/implot/implot.cpp',
    'thirdparty/implot/implot_demo.cpp',
    'thirdparty/implot/implot_items.cpp',
]
	
kobra_includes = [
	'/usr/include/ImageMagick-7',
	'thirdparty/freetype/include',
	'thirdparty/glm',
	'thirdparty/optix',
	'thirdparty/termcolor/include',
	'thirdparty/tinyfiledialogs',
    '/usr/include/opencv4',
    'thirdparty',
    'thirdparty/imgui',
]

kobra_libraries = [
    'glfw',
	'vulkan',
	'assimp',
    'pthread',
	'freetype',
    'glslang',
    'SPIRV',
    'OSDependent',
    'OGLCompiler',
	'opencv_core',
	'opencv_videoio',
	'opencv_imgcodecs',
	'opencv_imgproc',
	'Magick++-7.Q16HDRI', # TODO: find header and libraries...
	'MagickWand-7.Q16HDRI',
	'MagickCore-7.Q16HDRI'
]

kobra_dir = os.path.abspath(os.path.dirname(__file__))

# Warning suppression
def suppress_nvcc_warnings():
    nvcc_suppress = [20012, 20013, 20014]
    flags = ' -Xcudafe "'
    for code in nvcc_suppress:
        flags += ' --diag_suppress ' + str(code)
    return flags + '"'
