#ifndef PHYSICS_H_
#define PHYSICS_H_

// Standard headers
#include <vector>

// Engine headers
// #include "include/physics/collision_object.hpp"
// #include "include/physics/rigidbody.hpp"
#include "include/ui/line.hpp"
#include "include/transform.hpp"

#include <btBulletDynamicsCommon.h>

namespace mercury {

namespace physics {

// Bullet physics debugging
struct BulletDebugger : public btIDebugDraw {
	int dbmode;

	void drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) override {
		glm::vec3 from_glm {from.x(), from.y(), from.z()};
		glm::vec3 to_glm {to.x(), to.y(), to.z()};
		glm::vec3 color_glm {color.x(), color.y(), color.z()};

		ui::Line line(from_glm, to_glm, color_glm);
		line.draw(winman.cres.line_shader);
	}

	void drawContactPoint(const btVector3 &PointOnB, const btVector3 &normalOnB, btScalar distance, int lifeTime, const btVector3 &color) override {
		glm::vec3 point_glm {PointOnB.x(), PointOnB.y(), PointOnB.z()};
		glm::vec3 normal_glm {normalOnB.x(), normalOnB.y(), normalOnB.z()};
		glm::vec3 color_glm {color.x(), color.y(), color.z()};

		ui::Line line(point_glm, point_glm + normal_glm, color_glm);
		line.draw(winman.cres.line_shader);
	}

	void reportErrorWarning(const char *wstr) override {
		Logger::error() << " (BULLET) " << wstr << '\n';
	}

	void draw3dText(const btVector3 &location, const char *textString) override {}

	void setDebugMode(int debugMode) override {
		dbmode = debugMode;
	}

	int getDebugMode() const override {
		return dbmode;
	}
};

// TODO: collider classes, which has reference to transform?

// Physics daemon
class Daemon {
        // Bullet physics world stuff
        btDefaultCollisionConfiguration *	_cconfig;
        btCollisionDispatcher *			_dispatcher;
        btBroadphaseInterface *			_broadphase;
        btSequentialImpulseConstraintSolver *	_solver;
        btDiscreteDynamicsWorld *		_world;
        btIDebugDraw *				_debugger;

	// Array of collision objects
	std::vector <btCollisionShape *>	_cshapes;
public:
        // Constructor
        Daemon();

        // Adding collision bodies to the daemon
	// TODO: rename to add_rb?
	btRigidBody *add(float, Transform *, btCollisionShape *);

        // Run physics daemon
        void update(float);
};

}

}

#endif