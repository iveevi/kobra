#include "include/physics/collisionbody.hpp"

namespace mercury {

namespace physics {

CollisionBody::CollisionBody(Transform *t, Collider *c) :
                transform(t), collider(c) {}

}

}