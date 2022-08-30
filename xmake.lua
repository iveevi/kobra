add_includedirs("thirdparty")
add_includedirs("thirdparty/glm")
add_includedirs("thirdparty/freetype/include")
add_includedirs("thirdparty/optix")

target("experimental")
	set_kind("binary")
	add_files("experimental/main.cpp")
