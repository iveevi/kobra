#version 450

// Inputs
layout (location = 1) in vec2 in_uv;

// Outputs
layout(location = 0) out vec4 fragment;

// Hue to RGB conversion
vec4 hue_to_rgb(float hue)
{
	float red = 0.0;
	float green = 0.0;
	float blue = 0.0;
	if (hue >= 0.0 && hue < 60.0) {
		red = 1.0;
		green = hue/60.0;
		blue = 0.0;
	} else if(hue >= 60.0 && hue < 120.0) {
		red = 1.0 - (hue - 60.0)/60.0;
		green = 1.0;
		blue = 0.0;
	} else if(hue >= 120.0 && hue < 180.0) {
		red = 0.0;
		green = 1.0;
		blue = (hue - 120.0)/60.0;
	} else if(hue >= 180.0 && hue < 240.0) {
		red = 0.0;
		green = 1.0 - (hue - 180.0)/60.0;
		blue = 1.0;
	} else if(hue >= 240.0 && hue < 300.0) {
		red = (hue - 240.0)/60.0;
		green = 0.0;
		blue = 1.0;
	} else if(hue >= 300.0 && hue <= 360.0) {
		red = 1.0;
		green = 0.0;
		blue = 1.0 - (hue - 300.0)/60.0;
	}

	return vec4(red, green, blue, 1);
}

// Main function
void main()
{
	float hue = in_uv.y * 360.0;
	fragment = hue_to_rgb(hue);
}
