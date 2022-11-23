import os

# Definitions
kobra_sources = [
	'source/*',
	'source/io/*',
	'source/layers/*.cpp',
	'source/layers/basilisk.cu',
	'source/layers/denoiser.cu',
	'source/layers/optix_tracer.cu',
	'source/asmodeus/backend.cu',
	'source/arbok/*'
]
	
kobra_includes = [
    'thirdparty',
	'thirdparty/freetype/include',
	'thirdparty/glm',
	'thirdparty/optix',
	'thirdparty/termcolor/include',
	'thirdparty/tinyfiledialogs',
    '/usr/include/opencv4',
	'/usr/include/ImageMagick-7'
]

kobra_libraries = [
    'glfw',
	'vulkan',
	'assimp',
    'pthread',
	'freetype',
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
