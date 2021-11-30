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

btRigidBody *Daemon::add(CollisionObject cobj)
{
        // Create the transform
	// TODO: a method to convert transform to btTransform
	btTransform gtr;
	
	gtr.setIdentity();
	gtr.setOrigin(cobj.tr->translation);
	gtr.setRotation(cobj.tr->orient);

        // Create the rigid body
	btScalar btmass(cobj.mass);

        // Check dynamic or static
	bool is_dynamic = (btmass != 0.0f);

	btVector3 loc_inert(0, 0, 0);
	if (is_dynamic)
		cobj.shape->calculateLocalInertia(btmass, loc_inert);

        // Set motion state stuff
	btDefaultMotionState* mstate = new btDefaultMotionState(gtr);
	btRigidBody::btRigidBodyConstructionInfo rb_info(btmass, mstate, cobj.shape, loc_inert);
	btRigidBody* body = new btRigidBody(rb_info);

        // Add the rigid body to the world
	_world->addRigidBody(body);
        
	// Add collision object
	cobj.body = body;
	_cobjs.push_back(cobj);

	// TOOD: stop returning
	return body;
}

void Daemon::update(float delta_t)
{
	_world->stepSimulation(delta_t, 10);
	_world->debugDrawWorld();

	// TODO: a method to convert btTransform to our Transform
	for (const auto &cobj : _cobjs) {
		btVector3 p = cobj.body->getWorldTransform().getOrigin();
		btQuaternion q = cobj.body->getWorldTransform().getRotation();
		cobj.tr->translation = {p.getX(), p.getY(), p.getZ()};
		cobj.tr->orient = {q.getW(), q.getX(), q.getY(), q.getZ()};
	}
}

}

}