#ifndef HANDLER_H_
#define HANDLER_H_

namespace mercury {

class Handler {
	// Pass the pointer to the data
	// (not on the heap) and check
	// size_t value at the pointer
	// as "inheritance"
	virtual void call(size_t *) const = 0;
public:
	inline void run(size_t *data) {
		return call(data);
	}

};

// TODO: should we cast to reduce casting overhead within functions?
template <class T>
class MemFtnHandler : public Handler {
public:
	using MemFtn = void (T::*)(size_t *);
private:
	T *	_obj;
	MemFtn	_mftn;
public:
	MemFtnHandler(T *instance, MemFtn mftn)
		: _obj(instance), _mftn(mftn) {}

	void call(size_t *data) const override {
		(_obj->*_mftn)(data);
	}
};

}

#endif
