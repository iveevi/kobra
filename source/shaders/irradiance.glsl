#version 450

layout (binding = 0) uniform sampler2D environment_map;
layout (binding = 1) writeonly uniform image2D irradiance_map;

// Push constants
layout (push_constant) uniform PushConstants {
	float roughness;
	int width;
	int height;
} push_constants;

const float M_PI = 3.1415926535897932384626433832795;

vec3 uv_to_dir(vec2 uv)
{
	float phi = 2.0 * M_PI * (1 - uv.x) - M_PI/2.0;
	float theta = M_PI * uv.y;

	return vec3(
		sin(theta) * cos(phi),
		sin(theta) * sin(phi),
		cos(theta)
	);
}

vec2 dir_to_uv(vec3 dir)
{
	float phi = atan(dir.x, dir.y);
	float theta = asin(dir.z);
	vec2 uv = vec2(phi, theta) / vec2(2.0 * M_PI, M_PI) + vec2(0.5, 0.5);
	uv.y = 1.0 - uv.y;
	return uv;
}

vec3 rotate(vec3 s, vec3 n)
{
	vec3 w = n;
	vec3 a = vec3(0.0f, 1.0f, 0.0f);

	if (abs(dot(w, a)) > 0.999f)
		a = vec3(0.0f, 0.0f, 1.0f);

	if (abs(dot(w, a)) > 0.999f)
		a = vec3(0.0f, 0.0f, 1.0f);

	vec3 u = normalize(cross(w, a));
	vec3 v = normalize(cross(w, u));

	return u * s.x + v * s.y + w * s.z;
}

// Van Der Corpus sequence
// @see http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float vdcSequence(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// Hammersley sequence
// @see http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
vec2 hammersleySequence(uint i, uint N)
{
    return vec2(float(i) / float(N), vdcSequence(i));
}

// GGX NDF via importance sampling
vec3 importanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;

	float phi = 2.0 * M_PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (alpha2 - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	// from spherical coordinates to cartesian coordinates
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;

	// from tangent-space vector to world-space sample vector
	vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent   = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);

	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}

// Normal Distribution
float d_ggx(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return alpha2 / (M_PI * denom * denom); 
}

void main()
{
	// Get the uv coordinates of the pixel
	ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
	vec2 uv = vec2(coords) / vec2(push_constants.width, push_constants.height);

	vec3 N = uv_to_dir(uv);

	vec3 R = N;
	vec3 V = N;

	float roughness = push_constants.roughness;

	float totalWeight = 0.0;   
	vec3 prefilteredColor = vec3(0.0);
	int totalSamples = max(int(250.0f * roughness), 1);
	for(uint i = 0u; i < totalSamples; ++i) {
		// generate sample vector towards the alignment of the specular lobe
		vec2 Xi = hammersleySequence(i, totalSamples);
		vec3 H = importanceSampleGGX(Xi, N, roughness);
		float dotHV = dot(H, V);
		vec3 L = normalize(2.0 * dotHV * H - V);

		float dotNL = max(dot(N, L), 0.0);
		if(dotNL > 0.0) {
			float dotNH = max(dot(N, H), 0.0);
			dotHV = max(dotHV, 0.0);
			float D = d_ggx(dotNH, roughness);
			float pdf = D * dotNH / (4.0 * dotHV) + 0.0001; 
			vec2 uv2 = dir_to_uv(L);
			prefilteredColor += texture(environment_map, uv2).rgb * dotNL;
			totalWeight += dotNL;
		}
	}
	vec3 color = prefilteredColor / totalWeight;

	// Write the color to the irradiance map
	imageStore(irradiance_map, coords, vec4(color, 1) + vec4(0, 0, 0, 0.0));
}
