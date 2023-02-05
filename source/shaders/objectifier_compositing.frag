#version 450

// Only input is id
layout(location = 0) flat in uvec2 id;

// Push constants
layout (push_constant) uniform PushConstants {
	layout (offset = 200) uvec2 expected;
};

// Output is the id
layout(location = 0) out vec4 fragment;

void main()
{
	if (id == expected)
		fragment = vec4(1, 1, 0, 0.25);
	else
		discard;
}
