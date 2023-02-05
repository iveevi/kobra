#version 450

// Only input is id
layout(location = 0) flat in uvec2 id;

// Output is the id
layout(location = 0) out uvec2 out_id;

void main()
{
	out_id = id + 1;
}
