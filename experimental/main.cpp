#include <cassert>
#include <stack>

#include "lexer.hpp"
#include "value.hpp"
#include "instruction.hpp"

std::string source = R"(
# Defining components:
200 * 16 + 10.0/2.5 - 1
x = 200 * 16 + 10.0/2.5 - 1
y = 200.0
z = "Hello world!"
w = 'Hello world!'
)";

using _stack = std::vector <_value>;

machine m;

using nabu::parser::rd::alias;
using nabu::parser::rd::option;
using nabu::parser::rd::repeat;
using nabu::parser::rd::grammar;
using nabu::parser::rd::grammar_action;

using p_string = option <double_str, single_str>;
using primitive = option <p_int, p_float, p_string>;

using factor = option <primitive, identifier>;
using mul_factor = alias <multiply, factor>;
using div_factor = alias <divide, factor>;

using term = alias <factor, repeat <option <mul_factor, div_factor>>>;
using add_term = alias <plus, term>;
using sub_term = alias <minus, term>;

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
#define define_action(...)							\
	template <>								\
	struct grammar_action <__VA_ARGS__> {					\
		enable(true);							\
		static void action(parser::lexicon, parser::Queue &);		\
	};									\
										\
	void grammar_action <__VA_ARGS__>					\
		::action(parser::lexicon lptr, parser::Queue &q)

define_action(p_int)
{
	_value v;
	v.type = _value::Type::eInt;
	v.data = get <int> (lptr);

	push(m, v);
	push(m, _instruction::Type::ePushTmp);
}

define_action(p_float)
{
	_value v;
	v.type = _value::Type::eFloat;
	v.data = get <float> (lptr);

	push(m, v);
	push(m, _instruction::Type::ePushTmp);
}

define_action(p_string)
{
	_value v;
	v.type = _value::Type::eString;
	v.data = get <std::string> (lptr);

	push(m, v);
	push(m, _instruction::Type::ePushTmp);
}

define_action(add_term)
{
	push(m, _instruction::Type::eAdd);
}

define_action(sub_term)
{
	push(m, _instruction::Type::eSub);
}

define_action(mul_factor)
{
	push(m, _instruction::Type::eMul);
}

define_action(div_factor)
{
	push(m, _instruction::Type::eDiv);
}

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

	using mul_factor = alias <multiply, factor>;
	using div_factor = alias <divide, factor>;

	using g_input = grammar <term, repeat <option <add_term, sub_term>>>;
	// using g_input = grammar <alias <p_int, multiply, p_int>>;
	g_input::value(q);
	dump(m);

	exec(m);
	dump(m);

#endif

	return 0;
}
