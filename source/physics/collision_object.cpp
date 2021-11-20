#include "include/physics/collision_object.hpp"

namespace mercury {

namespace physics {

CollisionObject::CollisionObject(Collider *co, Type tp)
                : transform(co->transform), collider(co), type(tp) {}

CollisionObject::CollisionObject(Transform *tf, Collider *co, Type tp)
                : transform(tf), collider(co), type(tp) {}

}

}