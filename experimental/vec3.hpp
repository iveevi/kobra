#ifndef VEC3_H_
#define VEC3_H_

// Vector structure
struct vec3 {
        float x;
        float y;
        float z;
};

// Operators
inline vec3 operator+(const vec3& a, const vec3& b) {
        vec3 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        return c;
}

inline vec3 operator-(const vec3& a, const vec3& b) {
        vec3 c;
        c.x = a.x - b.x;
        c.y = a.y - b.y;
        c.z = a.z - b.z;
        return c;
}

#endif
