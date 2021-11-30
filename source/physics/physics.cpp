#include "include/physics/physics.hpp"

namespace mercury {

namespace physics {

Daemon::Daemon()
{
        // Initialize Bullet
        _cconfig = new btDefaultCollisionConfiguration();
        _dispatcher = new btCollisionDispatcher(_cconfig);
        _broadphase = new btDbvtBroadphase();
        _solver = new btSequentialImpulseConstraintSolver();
        _world = new btDiscreteDynamicsWorld(_dispatcher, _broadphase, _solver, _cconfig);
        _debugger = new BulletDebugger();

	// Set debugger
	_debugger->setDebugMode(btIDebugDraw::DBG_DrawWireframe);
	_world->setDebugDrawer(_debugger);
}

btRigidBody *Daemon::add(float mass, Transform *transform, btCollisionShape *cshape)
{
        // Add collision shape
	_cshapes.push_back(cshape);

        // Create the transform
	btTransform gtr;
	
	gtr.setIdentity();
	gtr.setOrigin(transform->translation);
	gtr.setRotation(transform->orient);

        // Create the rigid body
	btScalar btmass(mass);

        // Check dynamic or static
	bool is_dynamic = (btmass != 0.0f);

	btVector3 loc_inert(0, 0, 0);
	if (is_dynamic)
		cshape->calculateLocalInertia(mass, loc_inert);

        // Set motion state stuff
	btDefaultMotionState* mstate = new btDefaultMotionState(gtr);
	btRigidBody::btRigidBodyConstructionInfo rb_info(btmass, mstate, cshape, loc_inert);
	btRigidBody* body = new btRigidBody(rb_info);

        // Add the rigid body to the world
	_world->addRigidBody(body);

	// TOOD: stop returning
	return body;
}

void Daemon::update(float delta_t)
{
	_world->stepSimulation(delta_t, 10);
	_world->debugDrawWorld();
}

}

}