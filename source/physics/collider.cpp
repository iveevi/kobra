// Standard headers
#include <algorithm>

// Engine headers
#include "include/common.hpp"
#include "include/physics/collider.hpp"

// TODO: remove later
#include "include/model.hpp"
#include "include/mesh/cuboid.hpp"
#include "include/mesh/sphere.hpp"

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

// Get the normals for each face
struct FaceInfo {
	glm::uvec3		face;
	glm::vec3		normal;
	float			distance;

	// All normals for repairing the polytope
	std::vector <glm::vec3>	normals;
};

FaceInfo closest_face(const Collider::Vertices &vertices, const std::vector <glm::uvec3> &faces)
{
	// List of all normals
	std::vector <glm::vec3> normals;

	// Minimum info
	glm::uvec3 minf = {0, 0, 0};
	glm::vec3 minn = {0.0f, 0.0f, 0.0f};
	float mind = std::numeric_limits <float> ::max();

	for (const glm::uvec3 &face : faces) {
		glm::vec3 a = vertices[face[0]];
		glm::vec3 b = vertices[face[1]];
		glm::vec3 c = vertices[face[2]];

		glm::vec3 ab = b - a;
		glm::vec3 ac = c - a;
		glm::vec3 n = glm::normalize(glm::cross(ab, ac));

		// Flip the normal if necessary
		float d = glm::dot(n, a);
		if (d < 0) {
			n = -n;
			d = -d;
		}

		if (d < mind) {
			mind = d;
			minf = face;
			minn = n;
		}

		normals.push_back(n);
	}

	return {minf, minn, mind, normals};
}

// Check if a face faces a vertex
bool faces_vertex(const glm::vec3 face[3], const glm::vec3 &normal, const glm::vec3 &vertex)
{
	// NOTE: the first vertex was always used to compute the normal
	return glm::dot(normal, vertex - face[0]) > 0.0f;
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

glm::vec3 project_origin(const glm::vec3 vert, const glm::vec3 &n)
{
	glm::vec3 v = -vert;
	float d = glm::dot(n, v);
	return -d * n;
}

glm::vec3 barycentric(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const glm::vec3 &p)
{
	glm::vec3 v0 = b - a;
	glm::vec3 v1 = c - a;
	glm::vec3 v2 = p - a;

	float d00 = glm::dot(v0, v0);
	float d01 = glm::dot(v0, v1);
	float d11 = glm::dot(v1, v1);
	
	float d20 = glm::dot(v2, v0);
	float d21 = glm::dot(v2, v1);
	
	float denom = d00 * d11 - d01 * d01;

	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;

	return glm::vec3 {1.0f - v - w, v, w};
}

glm::vec3 tri_support(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const glm::vec3 &n)
{
	float d0 = glm::dot(n, a);
	float d1 = glm::dot(n, b);
	float d2 = glm::dot(n, c);

	if (d0 > d1) {
		if (d2 > d0)
			return c;

		return a;
	}

	if (d2 > d1)
		return c;
	
	return b;
}

MTV mtv(Simplex &simplex, Collider *a, Collider *b, rendering::Daemon *rdam, Shader *shader)
{
	Collider::Vertices polytope = simplex.vertices();
	std::vector <glm::uvec3> faces {
		{0, 1, 2},
		{0, 3, 1},
		{0, 2, 3},
		{1, 3, 2}
	};

	// TODO: set a loop counter a backup notifier for infinite loops
	Logger::warn("MTV LOOP:");

	float dprev = -1.0f;
	while (true) {
		// Get the closest face in the polytope
		FaceInfo fi = closest_face(polytope, faces);

		// Support in the direction of the normal of the closest face
		glm::vec3 s = support(fi.normal, a, b);
		
		// Compare the distance between the projection and the support
		float d = glm::dot(s, fi.normal);
		Logger::notify() << "\tdistance = " << d << std::endl;
		Logger::notify() << "\tfi.distance = " << fi.distance << std::endl;
		
		// Success, now returns the MTV
		if (fabs(d - fi.distance) < 1e-4f) {
			Logger::notify() << "MTV SUCCESS (MAIN)" << std::endl;
			glm::vec3 t = project_origin(s, fi.normal);
			
			glm::vec3 pa = polytope[fi.face.x];
			glm::vec3 pb = polytope[fi.face.y];
			glm::vec3 pc = polytope[fi.face.z];

			glm::vec3 sa1 = a->support(pa);
			glm::vec3 sa2 = a->support(pb);
			glm::vec3 sa3 = a->support(pc);

			/* Annotate points
			SVA3 *p1 = new SVA3(mesh::wireframe_sphere(sa1, 0.1f));
			SVA3 *p2 = new SVA3(mesh::wireframe_sphere(sa2, 0.1f));
			SVA3 *p3 = new SVA3(mesh::wireframe_sphere(sa3, 0.1f));
			p1->color = {0, 1, 0};
			p2->color = {0, 1, 0};
			p3->color = {0, 1, 0};

			rdam->add(p1, shader);
			rdam->add(p2, shader);
			rdam->add(p3, shader); */

			glm::vec3 bary = barycentric(pa, pb, pc, t);

			glm::vec3 ca = sa1 * bary.x + sa2 * bary.y + sa3 * bary.z;			

			return {t, ca, {0, 0, 0}};
		}

		// Check if entering loop
		if (fabs(d - dprev) < 1e-4f) {	// TODO: store epsilon
			Logger::notify() << "MTV SUCCESS (SECONDARY)" << std::endl;
			glm::vec3 t = project_origin(s, fi.normal); // Just return it
			glm::vec3 ca = a->support(-fi.normal);
			glm::vec3 cb = b->support(fi.normal);
			return {t, ca, cb};
		}

		expand_polytope(polytope, faces, fi.normals, s);

		// Set previous distances
		dprev = d;
	}

	return {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
}

// Intersection between two colliders
Collision intersects(Collider *a, Collider *b, rendering::Daemon *rdam, Shader *shader)
{
	// TODO: should aabb be cached?
	bool aabb_test = intersects(a->aabb(), b->aabb());

	// Check AABB first
	if (!aabb_test)
		return Collision {.colliding = false};

	// Now do GJK
	Simplex simplex;
	if (gjk(simplex, a, b)) {
		// Get centroid of simplex
		glm::vec3 centroid = simplex.centroid();

		MTV t = mtv(simplex, a, b, rdam, shader);
		// glm::vec3 as = a->support(-t);
		// glm::vec3 bs = b->support(t);

		// TODO: just return one collision point
		return Collision {
			.mtv = t.mtv,
			.contact_a = t.contact_a,
			.contact_b = t.contact_b,		// TODO: do we need bs?
			.colliding = true
		};
	}

	// Now onto more complicated tests
	return Collision {.colliding = false};
}


}

}