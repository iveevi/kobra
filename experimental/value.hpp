#ifndef VALUE_H_
#define VALUE_H_

// Standard headers
#include <variant>
#include <string>
#include <stdexcept>

struct _value {
	enum class Type {
		eInt, eFloat, eBool, eString,
		eList, eDictionary, eFuncion,
		eVec2, eVec3, eVec4, eStruct
	} type;

	static constexpr const char *type_str[] = {
		"int", "float", "bool", "string",
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

inline const char *str(_value::Type t)
{
	return _value::type_str[static_cast <int> (t)];
}

inline std::string str(const _value &v)
{
	// TODO: print type only in debug mode
	std::string out = "(type: ";
	out += str(v.type);

	switch (v.type) {
	case _value::Type::eInt:
		out += ", value: " + std::to_string(v.get <int> ());
		break;
	case _value::Type::eFloat:
		out += ", value: " + std::to_string(v.get <float> ());
		break;
	case _value::Type::eBool:
		out += ", value: " + std::to_string(v.get <bool> ());
		break;
	case _value::Type::eString:
		out += ", value: " + v.get <std::string> ();
		break;
	default:
		break;
	}

	return out + ")";
}

// Operations
template <class T>
struct enum_type {
	static constexpr _value::Type type = _value::Type::eInt;
};

#define define_enum_type(T, e)						\
	template <>							\
	struct enum_type <T> {						\
		static constexpr _value::Type type = _value::Type::e;	\
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
			str(lhs.type) + ", " + str(rhs.type)
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
			str(lhs.type) + ", " + str(rhs.type)
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
		return _value(enum_type <float> ::type, v_float_a * v_int_b);

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
			str(lhs.type) + ", " + str(rhs.type)
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
			str(lhs.type) + ", " + str(rhs.type)
	);
}

#endif
