#ifndef KOBRA_ARBOK_VALUE_H_
#define KOBRA_ARBOK_VALUE_H_

// Standard headers
#include <variant>
#include <string>
#include <stdexcept>
#include <map>
#include <vector>

#include <iostream>

namespace kobra {

namespace arbok {

// Forward declarations
struct _value;

// Value types
enum Type : int {
	eGeneric, eVoid, eInt, eFloat,
	eBool, eString, eList,
	eDictionary, eFuncion,
	eVec2, eVec3, eVec4, eStruct,
	eStructId
};

// String versions of types
static constexpr const char *type_str[] = {
	"__value__", "void", "int", "float",
	"bool", "string",
	"list", "dictionary", "function",
	"vec2", "vec3", "vec4", "struct"
};

// TODO: conserve some memory by sharing addressesand member types variables
struct _struct {
	std::string name;
	std::map <std::string, int> addresses;

	// If _struct instance is used as a type
	// placeholder, the members is the default
	// value for each member
	std::vector <_value> members;
	std::vector <Type> member_types;
};

// Forward declarations
struct _struct;

struct _value {
	// Actual data
	using data_t = std::variant <int, float, bool, std::string, _struct>;

	Type type;
	data_t data;

	// Default constructor
	_value() : type(Type::eInt), data(0) {}

	// Proper constructor
	_value(Type t, data_t d)
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

inline _struct make_struct(const std::string &name,
		const std::vector <std::string> &member_names,
		const std::vector <Type> &member_types)
{
	_struct s;
	s.name = name;
	s.member_types = member_types;

	// Addresses: first member is at 0
	for (int i = 0; i < member_names.size(); i++) {
		s.addresses[member_names[i]] = i;
		s.members.push_back(_value {member_types[i], 0});
	}

	return s;
}

// Methods
std::string str(Type, const _value &);
std::string str(const _value &);
std::string info(const _value &);

inline std::string str(const _struct &s, bool verbose = false)
{
	std::string ret = s.name + " {";

	if (verbose) {
		for (auto pr : s.addresses) {
			ret += pr.first + ": " + std::to_string(pr.second);
			ret += " (" + str(s.member_types[pr.second], s.members[pr.second]) + ")";
			ret += " = " + str(s.members[pr.second]);
			if (pr.first != s.addresses.rbegin()->first)
				ret += ", ";
		}
	}

	ret += "}";
	return ret;
}

// Arithmetic operations
_value operator+(const _value &, const _value &);
_value operator-(const _value &, const _value &);
_value operator*(const _value &, const _value &);
_value operator/(const _value &, const _value &);
_value operator%(const _value &, const _value &);

// Boolean operations
_value operator==(const _value &, const _value &);
_value operator!=(const _value &, const _value &);

// Comparison operations
_value operator<(const _value &, const _value &);
_value operator>(const _value &, const _value &);
_value operator<=(const _value &, const _value &);
_value operator>=(const _value &, const _value &);

}

}

#endif
