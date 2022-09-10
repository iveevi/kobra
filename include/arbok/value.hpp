#ifndef KOBRA_ARBOK_VALUE_H_
#define KOBRA_ARBOK_VALUE_H_

// Standard headers
#include <variant>
#include <string>
#include <stdexcept>

namespace kobra {

namespace arbok {

struct _value {
	enum Type : int {
		eGeneric, eVoid, eInt, eFloat,
		eBool, eString, eList,
		eDictionary, eFuncion,
		eVec2, eVec3, eVec4, eStruct,
		eVariadic
	} type;

	static constexpr const char *type_str[] = {
		"__value__", "void", "int", "float",
		"bool", "string",
		"list", "dictionary", "function",
		"vec2", "vec3", "vec4", "struct"
	};

	// Actual data
	std::variant <int, float, bool, std::string> data;

	// Default constructor
	_value() : type(Type::eInt), data(0) {}

	// Proper constructor
	_value(Type t, std::variant <int, float, bool, std::string> d)
			: type(t), data(d) {}

	// TODO: try to be conservative with copies...
	// Copy constructor
	_value(const _value &other) : type(other.type),
			data(other.data) {}

	// Copy assignment
	_value &operator=(const _value &other) {
		type = other.type;
		data = other.data;
		return *this;
	}

	// Move constructor
	_value(_value &&other) : type(other.type),
			data(std::move(other.data)) {}

	// Move assignment
	_value &operator=(_value &&other) {
		type = other.type;
		data = std::move(other.data);
		return *this;
	}

	// Easier get
	template <class T>
	const T &get() const {
		return std::get <T> (data);
	}
};

// Methods
std::string str(_value::Type);
std::string str(const _value &);
std::string info(const _value &);

// Arithmetic operations
_value operator+(const _value &, const _value &);
_value operator-(const _value &, const _value &);
_value operator*(const _value &, const _value &);
_value operator/(const _value &, const _value &);

}

}

#endif
