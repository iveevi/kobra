#ifndef GRAMMAR_H_
#define GRAMMAR_H_

// Stanard headers
#include <cassert>

// Engine headers
#include "lexer.hpp"
#include "instruction.hpp"

// TODO: some way to use multiple machines?
extern machine m;

// Using statements
using nabu::parser::rd::alias;
using nabu::parser::rd::epsilon;
using nabu::parser::rd::grammar_action;
using nabu::parser::rd::option;
using nabu::parser::rd::repeat;

// Primitive types
using p_string = option <double_str, single_str>;
using p_bool = option <k_true, k_false>;
using primitive = option <p_int, p_float, p_string, p_bool>;

// Variable is distinct from identifier
using variable = option <identifier>;

// Wholistic expression forward declaration
struct expression;

// Function calls
struct function_call {
	using production_rule = alias <
		identifier, lparen, repeat <
			option <alias <expression, comma>, expression>
		>, rparen>;
};

// Factors (can be multiplied, divided, etc.)
struct factor {
	using _parenthesized = alias <lparen, expression, rparen>;
	using production_rule = option <
		function_call,
		_parenthesized,
		primitive,
		variable
	>;
};

// Terms (can be added, subtracted, etc.)
struct term {
	struct _term;
	using _mul = alias <multiply, factor>;
	using _div = alias <divide, factor>;

	struct _term : public alias <
		option <_mul, _div>,
		option <_term, epsilon>
	> {};

	using production_rule = alias <factor, option <_term, epsilon>>;
};

// Expression wholistic
struct expression {
	struct _expression;
	using _add = alias <plus, term>;
	using _sub = alias <minus, term>;

	struct _expression : public alias <
		option <_add, _sub>,
		option <_expression, epsilon>
	> {};

	using production_rule = alias <term, option <_expression, epsilon>>;
};

// Register grammars for debugging
register(p_string)
register(primitive)

register(function_call)

register(factor);
register(factor::_parenthesized);

register(term);
register(term::_term);
register(term::_mul);
register(term::_div);

register(expression);
register(expression::_expression);
register(expression::_add);
register(expression::_sub);

// Actions for primitive types 
nabu_define_action(p_int)
{
	_value v;
	v.type = _value::Type::eInt;
	v.data = get <int> (lptr);

	push(m, {_instruction::Type::ePushTmp, (int) m.tmp.size()});
	push(m, v);
}

nabu_define_action(p_float)
{
	_value v;
	v.type = _value::Type::eFloat;
	v.data = get <float> (lptr);

	push(m, {_instruction::Type::ePushTmp, (int) m.tmp.size()});
	push(m, v);
}

nabu_define_action(p_string)
{
	_value v;
	v.type = _value::Type::eString;
	v.data = get <std::string> (lptr);

	push(m, {_instruction::Type::ePushTmp, (int) m.tmp.size()});
	push(m, v);
}

nabu_define_action(p_bool)
{
	_value v;
	v.type = _value::Type::eBool;
	v.data = get <bool> (lptr);

	push(m, {_instruction::Type::ePushTmp, (int) m.tmp.size()});
	push(m, v);
}

// Actions for operations
nabu_define_action(term::_mul)
{
	std::cout << "term::_mul" << std::endl;
	push(m, _instruction::Type::eMul);
}

nabu_define_action(term::_div)
{
	std::cout << "term::_div" << std::endl;
	push(m, _instruction::Type::eDiv);
}

nabu_define_action(expression::_add)
{
	std::cout << "expression::_add" << std::endl;
	push(m, _instruction::Type::eAdd);
}

nabu_define_action(expression::_sub)
{
	std::cout << "expression::_sub" << std::endl;
	push(m, _instruction::Type::eSub);
}

// Actions for semantics
nabu_define_action(variable)
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

nabu_define_action(function_call)
{
	vec v = get <vec> (lptr);
	std::string name = get <std::string> (v[0]);
	assert(v.size() == 4);

	v = get <vec> (v[2]);
	int nargs = v.size();
	for (auto &e : v)
		std::cout << e->str() << std::endl;
	if (m.functions.map_ext.count(name) > 0) {
		// External function
		int index = m.functions.map_ext[name];
		push(m, {_instruction::Type::eCallExt, index, nargs});
	} else {
		std::cout << "Unknown function: " << name << std::endl;
		throw std::runtime_error("Unknown function");
	}
}

/*
nabu_define_action(assignment)
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
} */

// TODO: error checking trips

#endif
