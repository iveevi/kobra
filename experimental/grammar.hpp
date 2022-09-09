#ifndef GRAMMAR_H_
#define GRAMMAR_H_

// Stanard headers
#include <cassert>

// Engine headers
#include "lexer.hpp"
#include "value.hpp"
#include "instruction.hpp"

// TODO: some way to use multiple machines?
extern machine m;

// Using statements
using nabu::parser::rd::alias;
using nabu::parser::rd::option;
using nabu::parser::rd::repeat;
using nabu::parser::rd::grammar_action;

// Primitive types
using p_string = option <double_str, single_str>;
using p_bool = option <k_true, k_false>;
using primitive = option <p_int, p_float, p_string, p_bool>;

// Variable is distinct from identifier
using variable = option <identifier>;

// Factors
using factor = option <primitive, variable>;
using mul_factor = alias <multiply, factor>;
using div_factor = alias <divide, factor>;

// Terms
using term = alias <factor, repeat <option <mul_factor, div_factor>>>;
using add_term = alias <plus, term>;
using sub_term = alias <minus, term>;

// Expression wholistic
using expression = alias <term, repeat <option <add_term, sub_term>>>;

// Types of statements
using assignment = alias <type, identifier, equals, expression>;

// Statement wholistic
using statement = option <assignment>;

// TODO: +error checks

// Wholistic grammar
using input = alias <statement>;

register(p_string)
register(primitive)
register(factor)
register(term)
register(expression)
register(assignment)

// register(input)

#define enable(b) static constexpr bool available = b

// TODO: is_of_type for token and lexvalue interaction
// Action for expression: push value to stack

// TODO: can be function since there is no partial specialization
#define define_action(...)							\
	template <>								\
	struct grammar_action <__VA_ARGS__> {					\
		enable(true);							\
		static void action(parser::rd::DualQueue &, const lexicon &);	\
	};									\
										\
	void grammar_action <__VA_ARGS__>					\
		::action(parser::rd::DualQueue &dq, const lexicon &lptr)

// Actions for expressions
define_action(p_int)
{
	_value v;
	v.type = _value::Type::eInt;
	v.data = get <int> (lptr);

	push(m, {_instruction::Type::ePushTmp, (int) m.tmp.size()});
	push(m, v);
}

define_action(p_float)
{
	_value v;
	v.type = _value::Type::eFloat;
	v.data = get <float> (lptr);

	push(m, {_instruction::Type::ePushTmp, (int) m.tmp.size()});
	push(m, v);
}

define_action(p_string)
{
	_value v;
	v.type = _value::Type::eString;
	v.data = get <std::string> (lptr);

	push(m, {_instruction::Type::ePushTmp, (int) m.tmp.size()});
	push(m, v);
}

define_action(p_bool)
{
	_value v;
	v.type = _value::Type::eBool;
	v.data = get <bool> (lptr);

	push(m, {_instruction::Type::ePushTmp, (int) m.tmp.size()});
	push(m, v);
}

define_action(add_term)
{
	std::cout << "add: " << lptr->str() << std::endl;
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

// Actions for semantics
define_action(variable)
{
	// Get identifier
	std::string id = get <std::string> (lptr);
	if (m.variables.map.count(id) == 0) {
		std::cout << "Error: variable '" << id << "' not found" << std::endl;
		return;
	}

	// Push variable to stack
	int addr = m.variables.map[id];
	push(m, {_instruction::Type::ePushVar, addr});
}

define_action(assignment)
{
	// Get identifier
	vec v = get <vec> (lptr);
	assert(v.size() == 4);

	// Get name and type
	_value::Type type = get <_value::Type> (v[0]);
	std::string id = get <std::string> (v[1]);

	// Check if the symbol is being redefined with a different type
	machine::Frame &f = m.variables;
	if (f.map.count(id) > 0) {
		int addr = f.map[id];
		_value::Type old_type = f.types[addr];
		
		if (old_type != type) {
			// TODO: and then show line number, etc
			std::cerr << "redefinition of '" << id
				<< "' with a different type" << std::endl;
			exit(1);
		}
	}

	std::cout << "Storing into " << id << std::endl;

	// Make space in the machine
	int addr = m.variables.add(id, type);

	// Push a store instruction
	push(m, {_instruction::Type::eStore, addr});
}

// Error checking trips
// define_action() {}

#endif
