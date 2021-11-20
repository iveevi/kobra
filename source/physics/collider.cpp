// Standard headers
#include <algorithm>

// Engine headers
#include "include/common.hpp"
#include "include/physics/collider.hpp"
#include "include/mesh/cuboid.hpp"

namespace mercury {

namespace physics {

// Constructors
Collider::Collider(Transform *tptr) : transform(tptr) {}

// AABB construction based on support vectors
AABB Collider::aabb() const
{
	// Axes
	static const glm::vec3 axes[] = {
		{1, 0, 0}, {0, 1, 0}, {0, 0, 1},
		{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}
	};

	// Find min and max on each axis
	glm::vec3 max_x = support(axes[0]);
	glm::vec3 max_y = support(axes[1]);
	glm::vec3 max_z = support(axes[2]);

	glm::vec3 min_x = support(axes[3]);
	glm::vec3 min_y = support(axes[4]);
	glm::vec3 min_z = support(axes[5]);

	// Center and size
	glm::vec3 center = {
		((max_x + min_x) / 2.0f).x,
		((max_y + min_y) / 2.0f).y,
		((max_z + min_z) / 2.0f).z
	};

	glm::vec3 size = {
		(max_x - min_x).x,
		(max_y - min_y).y,
		(max_z - min_z).z
	};

	// Return AABB
	return AABB {center, size};
}

// Box Collider
BoxCollider::BoxCollider(const glm::vec3 &s, Transform *tptr)
		: Collider(tptr), center({0, 0, 0}), size(s) {}

BoxCollider::BoxCollider(const glm::vec3 &c, const glm::vec3 &s, Transform *tptr)
		: Collider(tptr), center(c), size(s) {}

// Support method
glm::vec3 BoxCollider::support(const glm::vec3 &dir) const
{
	// TODO: need some way to cache the vertices
	glm::mat4 model = transform->model();
	glm::vec3 ncenter = glm::vec3(model * glm::vec4(center, 1.0f));
	glm::vec3 nsize = size * transform->scale/2.0f;			// Only consdider scale

	// All vertices
	glm::vec3 nright = glm::normalize(glm::vec3(model * glm::vec4 {1.0, 0.0, 0.0, 0.0}));
	glm::vec3 nup = glm::normalize(glm::vec3(model * glm::vec4 {0.0, 1.0, 0.0, 0.0}));
	glm::vec3 nforward = glm::normalize(glm::vec3(model * glm::vec4 {0.0, 0.0, 1.0, 0.0}));
	
	Collider::Vertices verts {
		ncenter + nright * nsize.x + nup * nsize.y + nforward * nsize.z,
		ncenter + nright * nsize.x + nup * nsize.y - nforward * nsize.z,
		ncenter + nright * nsize.x - nup * nsize.y + nforward * nsize.z,
		ncenter + nright * nsize.x - nup * nsize.y - nforward * nsize.z,
		ncenter - nright * nsize.x + nup * nsize.y + nforward * nsize.z,
		ncenter - nright * nsize.x + nup * nsize.y - nforward * nsize.z,
		ncenter - nright * nsize.x - nup * nsize.y + nforward * nsize.z,
		ncenter - nright * nsize.x - nup * nsize.y - nforward * nsize.z
	};
	
	// Loop through all vertices
	glm::vec3 vmax;

	float dmax = -std::numeric_limits <float> ::max();
	for (const glm::vec3 &v : verts) {
		float d = glm::dot(dir, v);

		if (d > dmax) {
			dmax = d;
			vmax = v;
		}
	}

	return vmax;
}

// TODO: use cres.basic
void BoxCollider::annotate(rendering::Daemon &rdam, Shader *shader) const
{
	static const glm::vec3 color = {1.0, 0.5, 1.0};

	glm::mat4 model = transform->model();
	glm::vec3 up = {0, 1, 0};
	glm::vec3 right = {1, 0, 0};

	SVA3 *box = new SVA3(mesh::wireframe_cuboid(center, size, up, right));
	box->color = color;

	rdam.add(box, shader, transform);
}

////////////////////////////////////////////////////////////////////////////////
//				Collider intersection			      //
////////////////////////////////////////////////////////////////////////////////

// TODO: clean up

// Support of minkowski difference
inline glm::vec3 support(const glm::vec3 &dir, const Collider *a, const Collider *b)
{
	return a->support(dir) - b->support(-dir);
}

// Check if vectors are in the same direction (TODO: put in linalg)
bool same_direction(const glm::vec3 &v1, const glm::vec3 &v2)
{
	return glm::dot(v1, v2) > 0;
}

// Simplex stages (TODO: should be these be Simplex methods?)
bool line_simplex(Simplex &simplex, glm::vec3 &dir)
{
	// Logger::warn() << "SIMPLEX-Line stage.\n";
	glm::vec3 a = simplex[0];
	glm::vec3 b = simplex[1];

	glm::vec3 ab = b - a;
	glm::vec3 ao = -a;

	if (same_direction(ab, ao)) {
		dir = glm::cross(glm::cross(ab, ao), ab);
	} else {
		simplex = {a};
		dir = ab;
	}

	return false;
}

bool triangle_simplex(Simplex &simplex, glm::vec3 &dir)
{
	// Logger::warn() << "SIMPLEX-Triangle stage.\n";
	glm::vec3 a = simplex[0];
	glm::vec3 b = simplex[1];
	glm::vec3 c = simplex[2];

	glm::vec3 ab = b - a;
	glm::vec3 ac = c - a;
	glm::vec3 ao = -a;

	glm::vec3 abc = glm::cross(ab, ac);

	if (same_direction(glm::cross(abc, ac), ao)) {
		if (same_direction(ac, ao)) {
			simplex = {a, c};
			dir = glm::cross(glm::cross(ac, ao), ac);
		} else {
			simplex = {a, b};
			return line_simplex(simplex, dir);
		}
	} else {
		if (same_direction(glm::cross(ab, abc), ao)) {
			simplex = {a, b};
			return line_simplex(simplex, dir);
		} else {
			if (same_direction(abc, ao)) {
				dir = abc;
			} else {
				simplex = {a, b, c};
				dir = -abc;
			}
		}
	}

	return false;
}

bool tetrahedron_simplex(Simplex &simplex, glm::vec3 &dir)
{
	// Logger::warn() << "SIMPLEX-Tetrahedron stage.\n";
	glm::vec3 a = simplex[0];
	glm::vec3 b = simplex[1];
	glm::vec3 c = simplex[2];
	glm::vec3 d = simplex[3];

	glm::vec3 ab = b - a;
	glm::vec3 ac = c - a;
	glm::vec3 ad = d - a;
	glm::vec3 ao = -a;

	glm::vec3 abc = glm::cross(ab, ac);
	glm::vec3 acd = glm::cross(ac, ad);
	glm::vec3 adb = glm::cross(ad, ab);

	if (same_direction(abc, ao)) {
		simplex = {a, b, c};
		// Logger::warn() << "\tSIMPLEX-Tetrahedron SUB-stage: abc.\n";
		return triangle_simplex(simplex, dir);
	}
		
	if (same_direction(acd, ao)) {
		simplex = {a, c, d};
		// Logger::warn() << "\tSIMPLEX-Tetrahedron SUB-stage: acd.\n";
		return triangle_simplex(simplex, dir);
	}
 
	if (same_direction(adb, ao)) {
		simplex = {a, d, b};
		// Logger::warn() << "\tSIMPLEX-Tetrahedron SUB-stage: adb.\n";
		return triangle_simplex(simplex, dir);
	}

	// Logger::warn() << "\tSIMPLEX-Tetrahedron Stage COMPLETED\n";

	return true;
}

// Update the simplex (TODO: method)
bool next_simplex(Simplex &simplex, glm::vec3 &dir)
{
	// Cases for each simplex size
	switch (simplex.size()) {
	case 2:
		return line_simplex(simplex, dir);
	case 3:
		return triangle_simplex(simplex, dir);
	case 4:
		return tetrahedron_simplex(simplex, dir);
	}

	return false;
}

bool gjk(Simplex &simplex, const Collider *a, const Collider *b)
{
	// Logger::notify() << "Inside GJK function.\n";

	// First direction and support
	glm::vec3 dir {1.0f, 0.0f, 0.0f};
	glm::vec3 s = support(dir, a, b); // support(dir, va, vb);
	simplex.push(s);

	// Next direction
	dir = -s;

	size_t i = 0;
	while (i++ < 100) {	// TODO: put max iterations as a constant
	// while (true) {	// TODO: why can we not loop forever?
		// Logger::notify() << "\tDirection = " << dir << "\n";
		
		// Support
		s = support(dir, a, b);
		// s = support(dir, va, vb);

		// Check for no intersection
		if (glm::dot(s, dir) <= 0.0f)
			return false;
		
		simplex.push(s);
		if (next_simplex(simplex, dir))
			return true;
	}

	// Should not get here
	// Logger::fatal_error("GJK failed to converge.");
	return false;
}

// EPA algorithm
glm::vec3 polytope_center(const Collider::Vertices &vertices)
{
	glm::vec3 center {0.0f, 0.0f, 0.0f};
	for (const glm::vec3 &v : vertices)
		center += v;
	return center / (float) vertices.size();
}

// Get the normals for each face
struct NormalInfo {
	glm::vec3 normal;
	glm::uvec3 face;
	float distance;

	std::vector <glm::vec3> nfaces;
};

NormalInfo face_normals(const Collider::Vertices &vertices, const std::vector <glm::uvec3> &faces)
{
	// List of all normals
	std::vector <glm::vec3> normals;

	// Minimum info
	glm::uvec3 minf = {0, 0, 0};
	glm::vec3 minn = {0.0f, 0.0f, 0.0f};
	float mind = std::numeric_limits <float> ::max();

	glm::vec3 center = polytope_center(vertices);
	for (const glm::uvec3 &face : faces) {
		glm::vec3 a = vertices[face[0]];
		glm::vec3 b = vertices[face[1]];
		glm::vec3 c = vertices[face[2]];

		glm::vec3 ab = b - a;
		glm::vec3 ac = c - a;
		glm::vec3 n = glm::cross(ab, ac);

		// Flip the normal if necessary
		if (glm::dot(n, a - center) < 0.0f)
			n = -n;

		float d = glm::dot(n, a);
		if (d < mind) {
			mind = d;
			minf = face;
			minn = n;
		}

		normals.push_back(n);
	}

	return {minn, minf, mind, normals};
}

// Check that the normals are facing the right direction (TODO: some linalg function)
bool check_normals(const Collider::Vertices &vertices, const std::vector <glm::uvec3> &faces, const std::vector <glm::vec3> &normals)
{
	glm::vec3 center = polytope_center(vertices);

	for (size_t i = 0; i < faces.size(); i++) {
		glm::vec3 sample = vertices[faces[i].x] - center;
		if (glm::dot(sample, normals[i]) < 0.0f)
			return false;
	}

	return true;
}

// Check if a face faces a vertex
bool faces_vertex(const glm::vec3 face[3], const glm::vec3 &normal, const glm::vec3 &vertex)
{
	// NOTE: the first vertex was always used to compute the normal
	return glm::dot(normal, vertex - face[0]) < 0.0f;
}

// Expand a polytope with the new vertex
void expand_polytope(Collider::Vertices &vertices,
		std::vector <glm::uvec3> &faces,
		const std::vector <glm::vec3> &normals,
		const glm::vec3 &svert)
{
	// Edge structure
	struct Edge {
		unsigned int a;
		unsigned int b;

		bool operator==(const Edge &other) const {
			return (a == other.a) && (b == other.b);
		}
	};

	// Edge list
	std::vector <Edge> edges;

	// Get array of edges in a face
	auto get_edges = [&] (const glm::uvec3 &face) {
		Edge e1 {face[0], face[1]};
		Edge e2 {face[1], face[2]};
		Edge e3 {face[2], face[0]};

		return std::vector <Edge> {e1, e2, e3};
	};

	// TODO: we are not going to remove vertices from
	// the polytope right now. Should this be considered?

	// Iterate over all faces
	for (size_t i = 0; i < faces.size(); i++) {
		// Face
		glm::vec3 vface[3] = {
			vertices[faces[i].x],
			vertices[faces[i].y],
			vertices[faces[i].z]
		};

		// Check if the face faces the new vertex
		if (!faces_vertex(vface, normals[i], svert))
			continue;

		// Get iterator at this posiiton and remove it
		auto it = faces.begin() + i;
		faces.erase(it);

		// Get edges
		glm::uvec3 f = faces[i];
		Edge iface[3] = {
			Edge {f.x, f.y},
			Edge {f.y, f.z},
			Edge {f.z, f.x}
		};
		
		// Check edges
		for (size_t i = 0; i < 3; i++) {
			Edge e = iface[i];

			auto itr = std::find(edges.begin(), edges.end(), e);
			if (itr == edges.end())
				edges.push_back(e);
			else
				edges.erase(itr);	
		}

		// Account for the shift in indices
		i--;
	}

	// Create the new triangles
	size_t svi = vertices.size();
	vertices.push_back(svert);
	for (const Edge &e : edges) {
		faces.push_back(
			{e.a, e.b, svi}
		);
	}
}

glm::vec3 mtv(Simplex &simplex, Collider *a, Collider *b)
{
	// static const size_t maxi = 10;

	// Minimum distance encountered
	//	deals with infinte loops
	//	from numerical instabilities
	struct {
		float		d = std::numeric_limits <float> ::max();
		glm::vec3	n = {0, 0, 0};
		size_t		i = 0;		// Number of iterations
	} min;

	// Logger::notify() << "MTV Algorithm.\n";

	Collider::Vertices polytope = simplex.vertices();
	std::vector <glm::uvec3> faces {
		{0, 1, 2},
		{0, 3, 1},
		{0, 2, 3},
		{1, 3, 2}
	};

	// TODO: set a loop counter a backup notifier for infinite loops
	size_t i = 0;
	while (true) {
		// One interation of EPA
		NormalInfo ninfo = face_normals(polytope, faces);
		bool checked = check_normals(polytope, faces, ninfo.nfaces);

		// Logger::notify() << "\tninfo.normal = " << ninfo.normal.x << ", " << ninfo.normal.y << ", " << ninfo.normal.z << "\n";
		// Logger::notify() << "\tninfo.distance = " << ninfo.distance << "\n";

		glm::vec3 svert = support(ninfo.normal, a, b);

		// Logger::notify() << "\tsvert = " << svert.x << ", " << svert.y << ", " << svert.z << "\n";
		
		float sdist = glm::dot(svert, ninfo.normal);
		
		float d = sdist - ninfo.distance;
		if (d < min.d) {
			min.d = d;
			min.n = ninfo.normal;
			min.i = 0;
		} else if (fabs(d - min.d) < 0.00001f) {	// Assuming that ninfo is the same
			min.i++;

			// If many iterations have passed, just return
			if (min.i > 10)
				return ninfo.distance * glm::normalize(ninfo.normal);
		}

		// Logger::notify() << "\tsdist = " << sdist << "\n";
		// Logger::notify() << "\tdifference = " << sdist - ninfo.distance << "\n\n";

		if (fabs(sdist - ninfo.distance) < 0.1f) {
			// Logger::notify() << "Terminate EPA, normal is = " << ninfo.normal << "\n";
			return ninfo.distance * glm::normalize(ninfo.normal);
		}

		expand_polytope(polytope, faces, ninfo.nfaces, svert);
		
		// if (i++ > maxi)
		//	Logger::fatal_error("MTV Algorithm has exceed [maxi] iterations");
	}

	return {0, 0, 0};
}

// Intersection between two colliders
Collision intersects(Collider *a, Collider *b)
{
	// TODO: should aabb be cached?
	bool aabb_test = intersects(a->aabb(), b->aabb());

	// Check AABB first
	if (!aabb_test)
		return {{}, false};

	// Now do GJK
	Simplex simplex;
	if (gjk(simplex, a, b))
		return {mtv(simplex, a, b), true};

	// Now onto more complicated tests
	return {{}, false};
}


}

}