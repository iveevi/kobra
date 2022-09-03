#ifndef INSTRUCTION_H_
#define INSTRUCTION_H_

// Standard headers
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include <queue>

// Engine headers
#include "value.hpp"

// Abstraction of the machine state
struct _instruction;

struct machine {
	std::vector <_value> stack;
	std::queue <_value> tmp;

	std::vector <_value> variables;
	std::map <std::string, int> addresses;

	std::vector <_instruction> instructions;

	uint32_t pc = 0;
};

// At most two operands
struct _instruction {
	enum class Type {
		ePushTmp, ePushVar, ePop, eStore,
		eAdd, eSub, eMul, eDiv, eMod,
	} type;

	static constexpr const char *type_str[] = {
		"push_tmp", "push_var", "pop", "store",
		"add", "sub", "mul", "div", "mod",
	};

	// Index of operands in stack
	int op1, op2;

	_instruction(Type t, int o1 = -1, int o2 = -1)
			: type(t), op1(o1), op2(o2) {}
};

inline std::string str(const _instruction &i)
{
	std::string out = "(type: ";
	out += _instruction::type_str[(int)i.type];
	out += ", op1: ";
	out += std::to_string(i.op1);
	out += ", op2: ";
	out += std::to_string(i.op2);
	out += ")";
	return out;
}

// Execution table
std::unordered_map <
	_instruction::Type,
	std::function <void (machine &, const _instruction &)>
> exec_table {
	{_instruction::Type::ePushTmp, [](machine &m, const _instruction &i) {
		m.stack.push_back(m.tmp.front());
		m.tmp.pop();
		m.pc++;
	}},

	{_instruction::Type::ePushVar, [](machine &m, const _instruction &i) {
		m.stack.push_back(m.variables[i.op1]);
		m.pc++;
	}},

	{_instruction::Type::ePop, [](machine &m, const _instruction &i) {
		m.stack.pop_back();
		m.pc++;
	}},

	{_instruction::Type::eStore, [](machine &m, const _instruction &i) {
		m.variables[i.op1] = m.stack.back();
		m.stack.pop_back();
		m.pc++;
	}},

	{_instruction::Type::eAdd, [](machine &m, const _instruction &i) {
		_value v1 = m.stack.back();
		m.stack.pop_back();

		_value v2 = m.stack.back();
		m.stack.pop_back();

		m.stack.push_back(v1 + v2);
		m.pc++;
	}},

	{_instruction::Type::eSub, [](machine &m, const _instruction &i) {
		_value v1 = m.stack.back();
		m.stack.pop_back();

		_value v2 = m.stack.back();
		m.stack.pop_back();

		m.stack.push_back(v2 - v1);
		m.pc++;
	}},

	{_instruction::Type::eMul, [](machine &m, const _instruction &i) {
		_value v1 = m.stack.back();
		m.stack.pop_back();

		_value v2 = m.stack.back();
		m.stack.pop_back();

		m.stack.push_back(v1 * v2);
		m.pc++;
	}},

	{_instruction::Type::eDiv, [](machine &m, const _instruction &i) {
		_value v1 = m.stack.back();
		m.stack.pop_back();

		_value v2 = m.stack.back();
		m.stack.pop_back();

		m.stack.push_back(v2/v1);
		m.pc++;
	}},
};

inline void push(machine &m, const _instruction &i)
{
	m.instructions.push_back(i);
}

inline void push(machine &m, const _value &v)
{
	m.tmp.push(v);
}

inline void dump(const machine &m)
{
	std::cout << "=== Machine Dump ===" << std::endl;
	std::cout << "Stack size: " << m.stack.size() << std::endl;
	for (int i = 0; i < m.stack.size(); i++)
		std::cout << "[" << i << "]: " << str(m.stack[i]) << std::endl;

	std::cout << "\nVariables: " << m.addresses.size() << std::endl;
	for (auto &p : m.addresses) {
		std::cout << p.first << " @ " << p.second
			<< " = " << str(m.variables[p.second]) << std::endl;
	}

	std::cout << "\nInstructions: " << m.instructions.size() << std::endl;
	for (int i = 0; i < m.instructions.size(); i++)
		std::cout << "[" << i << "]: " << str(m.instructions[i]) << std::endl;
}

inline void exec(machine &m)
{
	while (m.pc < m.instructions.size()) {
		const auto &i = m.instructions[m.pc];
		exec_table[i.type](m, i);
	}
}

#endif
