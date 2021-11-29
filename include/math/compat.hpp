#ifndef COMPAT_H_
#define COMPAT_H_

// GLM headers
#include <glm/glm.hpp>

// Bullet headers
#include <LinearMath/btVector3.h>
#include <LinearMath/btQuaternion.h>

namespace mercury {

struct vec3 {
	float x, y, z;

	// Constructors
	vec3() : x(0.0f), y(0.0f), z(0.0f) {}
	vec3(float x, float y, float z) : x(x), y(y), z(z) {}
	vec3(const glm::vec3& v) : x(v.x), y(v.y), z(v.z) {}
	vec3(const btVector3& v) : x(v.x()), y(v.y()), z(v.z()) {}

	// Conversion to other vector types
	operator glm::vec3() const {
		return glm::vec3(x, y, z);
	}

	operator btVector3() const {
		return btVector3(x, y, z);
	}

	// Vector-vector operators
	vec3 &operator+=(const vec3 &) {
		x += x;
		y += y;
		z += z;
		return *this;
	}

	vec3 &operator-=(const vec3 &) {
		x -= x;
		y -= y;
		z -= z;
		return *this;
	}

	vec3 &operator*=(float) {
		x *= x;
		y *= y;
		z *= z;
		return *this;
	}

	vec3 &operator/=(float) {
		x /= x;
		y /= y;
		z /= z;
		return *this;
	}
};

struct quat {
	float w, x, y, z;

	// Constructors
	quat() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}
	quat(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}
	quat(const glm::quat& q) : w(q.w), x(q.x), y(q.y), z(q.z) {}
	quat(const btQuaternion& q) : w(q.w()), x(q.x()), y(q.y()), z(q.z()) {}

	// Conversion to other quaternion types
	operator glm::quat() const {
		return glm::quat(w, x, y, z);
	}

	operator btQuaternion() const {
		return btQuaternion(x, y, z, w);
	}
};

}

#endif