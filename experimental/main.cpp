#include <cassert>
#include <cstring>
#include <stack>
#include <variant>

#include "lexer.hpp"

std::string source = R"(
# Defining components:
x = 200 * 16 + 10.0/2.5 - 1
y = 200.0
z = "Hello world!"
w = 'Hello world!'
)";

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

inline std::string str(const _value &v)
{
	std::string out = "(type: ";
	out += _value::type_str[(int)v.type];

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

using _stack = std::vector <_value>;

struct State {
	// Value stack
	_stack stack;

	// Map from identifier to value
	std::map <std::string, _value> variables;

	void push(const _value &v) {
		std::cout << "Pushing " << str(v) << std::endl;
		stack.push_back(v);
	}

	void dump() {
		std::cout << "=== State Dump ===" << std::endl;
		std::cout << "Stack size: " << stack.size() << std::endl;
		for (int i = 0; i < stack.size(); i++)
			std::cout << "[" << i << "]: " << str(stack[i]) << std::endl;

		std::cout << "\nVariables: " << variables.size() << std::endl;
		for (auto &p : variables)
			std::cout << p.first << ": " << str(p.second) << std::endl;
	}
} state;

using nabu::parser::rd::alias;
using nabu::parser::rd::option;
using nabu::parser::rd::repeat;
using nabu::parser::rd::grammar;
using nabu::parser::rd::grammar_action;

using p_string = option <double_str, single_str>;
using primitive = option <p_int, p_float, p_string>;
using factor = option <primitive, identifier>;
using term = alias <factor, multiply, factor>;
using expression = alias <factor>;
using statement = alias <identifier, equals, expression>;
using input = statement;

register(p_string)
register(primitive)
register(factor)
register(term)
register(expression)
register(statement)
// register(input)

#define enable(b) static constexpr bool available = b

// TODO: is_of_type for token and lexvalue interaction
// Action for expression: push value to stack
#define define_action(T)							\
	template <>								\
	struct grammar_action <T> {						\
		enable(true);							\
		static void action(parser::lexicon, parser::Queue &);		\
	};									\
										\
	void grammar_action <T>::action(parser::lexicon lptr, parser::Queue &q)

define_action(p_int)
{
	_value v;
	v.type = _value::Type::eInt;
	v.data = get <int> (lptr);
	state.push(v);
}

define_action(p_float)
{
	_value v;
	v.type = _value::Type::eFloat;
	v.data = get <float> (lptr);
	state.push(v);
}

define_action(p_string)
{
	_value v;
	v.type = _value::Type::eString;
	v.data = get <std::string> (lptr);
	state.push(v);
}

template <>
struct grammar_action <statement> {
	enable(true);
	static void action(parser::lexicon lptr, parser::Queue &) {
		// Assumption is that the stack has the value
		// to be assigned to the identifier
		vec lvec = get <vec> (lptr);
		state.dump();

		// Get the identifier
		// assert(lvec.size() == 3);
		std::string id = get <std::string> (lvec[0]);

		// Get value
		assert(state.stack.size() > 0);
		std::cout << "Assigning " << str(state.stack.back()) << " to " << id << std::endl;
		_value v = state.stack.back();
		state.stack.pop_back();

		// Set variable
		state.variables[id] = v;
	}
};

int main()
{
	using namespace nabu;

	parser::Queue q = parser::lexq <identifier> (source);

#if 0

	std::cout << "Queue size: " << q.size() << std::endl;
	while (!q.empty()) {
		parser::lexicon lptr = q.front();
		q.pop_front();

		if (lptr == nullptr) {
			std::cout << "nullptr" << std::endl;
			continue;
		}

		std::cout << "lexicon: " << lptr->str() << std::endl;
	}

#else

	using g_input = grammar <input>;
	while (g_input::value(q));
	state.dump();

#endif

	return 0;
}
