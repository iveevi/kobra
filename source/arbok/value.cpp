// Standard headers
#include <cmath>

// Engine headers
#include "../../include/arbok/value.hpp"

namespace kobra {

namespace arbok {

std::string str(Type t, const _value &v)
{
	if (t > Type::eStruct)
		return "eStruct";

	return type_str[static_cast <int> (t)];
}

std::string str(const _value &v)
{
	// TODO: print type only in debug mode
	switch (v.type) {
	case Type::eInt:
		return std::to_string(v.get <int> ());
		break;
	case Type::eFloat:
		return std::to_string(v.get <float> ());
		break;
	case Type::eBool:
		return v.get <bool> () ? "true" : "false";
		break;
	case Type::eString:
		return v.get <std::string> ();
		break;
	default:
		break;
	}

	// TODO: Check for str() overloads in the future

	// Return address of value
	char buf[1024];
	snprintf(buf, 1024, "<%s@%p>", str(v.type, v).c_str(), &v);
	return buf;
}

std::string info(const _value &v)
{
	// TODO: print type only in debug mode
	std::string out = "(type: ";
	out += str(v.type, v);
	out += ", value: " + str(v);
	return out + ")";
}

// Operation overload helpers
template <class T>
struct enum_type {
	static constexpr Type type = Type::eInt;
};

#define define_enum_type(T, e)						\
	template <>							\
	struct enum_type <T> {						\
		static constexpr Type type = Type::e;	\
	}

define_enum_type(int, eInt);
define_enum_type(float, eFloat);
define_enum_type(bool, eBool);
define_enum_type(std::string, eString);

// Non-strict overload ordering
template <class A, class B, bool = false>
inline bool overload(const _value &x, const _value &y, A &a, B &b)
{
	if (x.type == enum_type <A> ::type && y.type == enum_type <B> ::type) {
		a = x.get <A> ();
		b = y.get <B> ();
		return true;
	} else if (x.type == enum_type <B> ::type && y.type == enum_type <A> ::type) {
		a = y.get <A> ();
		b = x.get <B> ();
		return true;
	}

	return false;
}

// Strict overload ordering
template <class A, class B>
inline bool strict_overload(const _value &x, const _value &y, A &a, B &b)
{
	if (x.type == enum_type <A> ::type && y.type == enum_type <B> ::type) {
		a = x.get <A> ();
		b = y.get <B> ();
		return true;
	}

	return false;
}

// Arithmetic operations
_value operator+(const _value &lhs, const _value &rhs)
{
	// Valid types
	int v_int_a, v_int_b;
	float v_float_a, v_float_b;
	bool v_bool_a, v_bool_b;
	std::string v_string_a, v_string_b;

	// Overloads
	if (overload <int, int> (lhs, rhs, v_int_a, v_int_b))
		return _value(enum_type <int> ::type, v_int_a + v_int_b);

	else if (overload <int, float> (lhs, rhs, v_int_a, v_float_b))
		return _value(enum_type <float> ::type, v_int_a + v_float_b);

	else if (overload <float, float> (lhs, rhs, v_float_a, v_float_b))
		return _value(enum_type <float> ::type, v_float_a + v_float_b);

	else if (overload <bool, bool> (lhs, rhs, v_bool_a, v_bool_b))
		return _value(enum_type <bool> ::type, v_bool_a + v_bool_b);

	else if (overload <std::string, std::string> (lhs, rhs, v_string_a, v_string_b))
		return _value(enum_type <std::string> ::type, v_string_a + v_string_b);

	throw std::runtime_error(
		std::string("Undefined overload for +: ") +
		str(lhs.type, lhs) + ", " + str(rhs.type, rhs)
	);
}

_value operator-(const _value &lhs, const _value &rhs)
{
	// Valid types
	int v_int_a, v_int_b;
	float v_float_a, v_float_b;
	bool v_bool_a, v_bool_b;

	// Overloads
	if (strict_overload <int, int> (lhs, rhs, v_int_a, v_int_b))
		return _value(enum_type <int> ::type, v_int_a - v_int_b);

	else if (strict_overload <int, float> (lhs, rhs, v_int_a, v_float_b))
		return _value(enum_type <float> ::type, v_int_a - v_float_b);

	else if (strict_overload <float, int> (lhs, rhs, v_float_a, v_int_b))
		return _value(enum_type <float> ::type, v_float_a - v_int_b);

	else if (strict_overload <float, float> (lhs, rhs, v_float_a, v_float_b))
		return _value(enum_type <float> ::type, v_float_a - v_float_b);

	else if (strict_overload <bool, bool> (lhs, rhs, v_bool_a, v_bool_b))
		return _value(enum_type <bool> ::type, v_bool_a - v_bool_b);

	throw std::runtime_error(
		std::string("Undefined overload for -: ") +
		str(lhs.type, lhs) + ", " + str(rhs.type, rhs)
	);
}

_value operator*(const _value &lhs, const _value &rhs)
{
	// Valid types
	int v_int_a, v_int_b;
	float v_float_a, v_float_b;
	bool v_bool_a, v_bool_b;
	std::string v_string;

	// Overloads
	if (overload <int, int> (lhs, rhs, v_int_a, v_int_b))
		return _value(enum_type <int> ::type, v_int_a * v_int_b);

	else if (overload <int, float> (lhs, rhs, v_int_a, v_float_b))
		return _value(enum_type <float> ::type, v_int_a * v_float_b);

	else if (overload <float, float> (lhs, rhs, v_float_a, v_float_b))
		return _value(enum_type <float> ::type, v_float_a * v_float_b);

	else if (overload <bool, bool> (lhs, rhs, v_bool_a, v_bool_b))
		return _value(enum_type <bool> ::type, v_bool_a * v_bool_b);

	else if (overload <std::string, int> (lhs, rhs, v_string, v_int_b)) {
		std::string out;
		for (int i = 0; i < v_int_b; i++)
			out += v_string;
		return _value(enum_type <std::string> ::type, out);
	}

	throw std::runtime_error(
		std::string("Undefined overload for *: ") +
		str(lhs.type, lhs) + ", " + str(rhs.type, rhs)
	);
}

_value operator/(const _value &lhs, const _value &rhs)
{
	// Valid types
	int v_int_a, v_int_b;
	float v_float_a, v_float_b;
	bool v_bool_a, v_bool_b;

	// Overloads
	if (strict_overload <int, int> (lhs, rhs, v_int_a, v_int_b))
		return _value(enum_type <int> ::type, v_int_a / v_int_b);

	else if (strict_overload <int, float> (lhs, rhs, v_int_a, v_float_b))
		return _value(enum_type <float> ::type, v_int_a / v_float_b);

	else if (strict_overload <float, int> (lhs, rhs, v_float_a, v_int_b))
		return _value(enum_type <float> ::type, v_float_a / v_int_b);

	else if (strict_overload <float, float> (lhs, rhs, v_float_a, v_float_b))
		return _value(enum_type <float> ::type, v_float_a / v_float_b);

	else if (strict_overload <bool, bool> (lhs, rhs, v_bool_a, v_bool_b))
		return _value(enum_type <bool> ::type, v_bool_a / v_bool_b);

	throw std::runtime_error(
		std::string("Undefined overload for /: ") +
		str(lhs.type, lhs) + ", " + str(rhs.type, rhs)
	);
}

_value operator%(const _value &lhs, const _value &rhs)
{
	// Valid types
	int v_int_a, v_int_b;
	float v_float_a, v_float_b;

	// Overloads
	if (strict_overload <int, int> (lhs, rhs, v_int_a, v_int_b))
		return _value(enum_type <int> ::type, v_int_a % v_int_b);

	else if (strict_overload <int, float> (lhs, rhs, v_int_a, v_float_b))
		return _value(enum_type <float> ::type, std::fmod(float(v_int_a), v_float_b));

	else if (strict_overload <float, int> (lhs, rhs, v_float_a, v_int_b))
		return _value(enum_type <float> ::type, std::fmod(v_float_a, float(v_int_b)));

	else if (strict_overload <float, float> (lhs, rhs, v_float_a, v_float_b))
		return _value(enum_type <float> ::type, std::fmod(v_float_a, v_float_b));

	throw std::runtime_error(
		std::string("Undefined overload for %: ") +
		str(lhs.type, lhs) + ", " + str(rhs.type, rhs)
	);
}

// Boolean operators
_value operator==(const _value &lhs, const _value &rhs)
{
	// Valid types
	int v_int_a, v_int_b;
	float v_float_a, v_float_b;
	bool v_bool_a, v_bool_b;
	std::string v_string_a, v_string_b;

	// Overloads
	if (overload <bool, bool> (lhs, rhs, v_bool_a, v_bool_b))
		return _value(enum_type <bool> ::type, v_bool_a == v_bool_b);

	else if (overload <int, int> (lhs, rhs, v_int_a, v_int_b))
		return _value(enum_type <bool> ::type, v_int_a == v_int_b);

	else if (overload <int, float> (lhs, rhs, v_int_a, v_float_b))
		return _value(enum_type <bool> ::type, float(v_int_a) == v_float_b);

	else if (overload <float, float> (lhs, rhs, v_float_a, v_float_b))
		return _value(enum_type <bool> ::type, v_float_a == v_float_b);

	else if (overload <std::string, std::string> (lhs, rhs, v_string_a, v_string_b))
		return _value(enum_type <bool> ::type, v_string_a == v_string_b);

	throw std::runtime_error(
		std::string("Undefined overload for ==: ") +
		str(lhs.type, lhs) + ", " + str(rhs.type, rhs)
	);
}

_value operator!=(const _value &lhs, const _value &rhs)
{
	// Valid types
	int v_int_a, v_int_b;
	float v_float_a, v_float_b;
	bool v_bool_a, v_bool_b;
	std::string v_string_a, v_string_b;

	// Overloads
	if (overload <bool, bool> (lhs, rhs, v_bool_a, v_bool_b))
		return _value(enum_type <bool> ::type, v_bool_a != v_bool_b);

	else if (overload <int, int> (lhs, rhs, v_int_a, v_int_b))
		return _value(enum_type <bool> ::type, v_int_a != v_int_b);

	else if (overload <int, float> (lhs, rhs, v_int_a, v_float_b))
		return _value(enum_type <bool> ::type, float(v_int_a) != v_float_b);

	else if (overload <float, float> (lhs, rhs, v_float_a, v_float_b))
		return _value(enum_type <bool> ::type, v_float_a != v_float_b);

	else if (overload <std::string, std::string> (lhs, rhs, v_string_a, v_string_b))
		return _value(enum_type <bool> ::type, v_string_a != v_string_b);

	throw std::runtime_error(
		std::string("Undefined overload for !=: ") +
		str(lhs.type, lhs) + ", " + str(rhs.type, rhs)
	);
}

}

}
