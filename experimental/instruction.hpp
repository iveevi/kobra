#ifndef INSTRUCTION_H_
#define INSTRUCTION_H_

// Standard headers
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <vector>

// Engine headers
#include "value.hpp"

// Abstraction of the machine state
struct _instruction;

struct machine {
	// TODO: move to stack frame
	std::vector <_value> stack;
	std::vector <_value> tmp;

	// Stack frame
	struct Frame {
		std::vector <_value> mem;
		std::vector <_value::Type> types;
		std::map <std::string, int> map;

		int add(const std::string &name, _value::Type type) {
			int addr = mem.size();
			map[name] = addr;
			mem.push_back(_value());
			types.push_back(type);
			return addr;
		}
	} variables;

	std::vector <_instruction> instructions;

	uint32_t pc = 0;
};

void dump(const machine &);

inline _value pop(machine &m)
{
	_value v = m.stack.back();
	m.stack.pop_back();
	return v;
}

// At most two operands
struct _instruction {
	enum class Type {
		ePushTmp, ePushVar, ePop, eStore,
		eAdd, eSub, eMul, eDiv, eMod,
		eCjmp, eNcjmp, eJmp, eCall, eRet,
		eEnd
	} type;

	static constexpr const char *type_str[] = {
		"push_tmp", "push_var", "pop", "store",
		"add", "sub", "mul", "div", "mod",
		"cjmp", "ncjmp", "jmp", "call", "ret",
		"end"
	};

	// Operands
	std::variant <int, _value> op1;
	std::variant <int, _value> op2;

	_instruction(Type t,
			std::variant <int, _value> o1 = -1,
			std::variant <int, _value> o2 = -1)
			: type(t), op1(o1), op2(o2) {}
};

inline std::string str(const _instruction &i)
{
	std::string out = "(type: ";
	out += _instruction::type_str[(int)i.type];

	out += ", op1: ";
	if (std::holds_alternative <int> (i.op1))
		out += std::to_string(std::get <int> (i.op1));
	else
		out += str(std::get <_value> (i.op1));

	out += ", op2: ";
	if (std::holds_alternative <int> (i.op2))
		out += std::to_string(std::get <int> (i.op2));
	else
		out += str(std::get <_value> (i.op2));

	out += ")";
	return out;
}

// Execution table
std::unordered_map <
	_instruction::Type,
	std::function <void (machine &, const _instruction &)>
> exec_table {
	{_instruction::Type::ePushTmp, [](machine &m, const _instruction &i) {
		assert(std::holds_alternative <int> (i.op1));
		int addr = std::get <int> (i.op1);
		m.stack.push_back(m.tmp[addr]);
		m.pc++;
	}},

	{_instruction::Type::ePushVar, [](machine &m, const _instruction &i) {
		assert(std::holds_alternative <int> (i.op1));
		int addr = std::get <int> (i.op1);
		m.stack.push_back(m.variables.mem[addr]);
		m.pc++;
	}},

	{_instruction::Type::ePop, [](machine &m, const _instruction &i) {
		m.stack.pop_back();
		m.pc++;
	}},

	{_instruction::Type::eStore, [](machine &m, const _instruction &i) {
		assert(std::holds_alternative <int> (i.op1));
		int addr = std::get <int> (i.op1);

		_value v = m.stack.back();
		m.stack.pop_back();

		// Make sure types are matching
		if (v.type != m.variables.types[addr]) {
			std::cerr << "Cannot assign value of type "
				<< str(v.type) << " to type "
				<< str(m.variables.types[addr]) << std::endl;
			exit(1);
		}

		m.variables.mem[addr] = v;
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

	{_instruction::Type::eCjmp, [](machine &m, const _instruction &i) {
		// Check previous stack value,
		//   then jump if true
		_value v = pop(m);

		if (v.type != _value::Type::eBool) {
			std::cerr << "Conditional clauses must be of type bool" << std::endl;
			std::cerr << "\tgot: " << str(v) << std::endl;
			exit(1);
		}

		assert(std::holds_alternative <int> (i.op1));
		int addr = std::get <int> (i.op1);
		if (v.get <bool> ()) {
			m.pc = addr;
		} else {
			m.pc++;
		}
	}},

	{_instruction::Type::eNcjmp, [](machine &m, const _instruction &i) {
		// Check previous stack value,
		//   then jump if false
		_value v = pop(m);

		if (v.type != _value::Type::eBool) {
			std::cerr << "Conditional clauses must be of type bool" << std::endl;
			std::cerr << "\tncjmp got: " << str(v) << " @" << m.pc << std::endl;
			dump(m);
			exit(1);
		}

		assert(std::holds_alternative <int> (i.op1));
		int addr = std::get <int> (i.op1);
		
		if (!v.get <bool> ())
			m.pc = addr;
		else
			m.pc++;
	}},

	{_instruction::Type::eJmp, [](machine &m, const _instruction &i) {
		assert(std::holds_alternative <int> (i.op1));
		m.pc = std::get <int> (i.op1);
	}},

	{_instruction::Type::eEnd, [](machine &m, const _instruction &i) {
		m.pc++;
	}}
};

inline void push(machine &m, const _instruction &i)
{
	m.instructions.push_back(i);
}

inline void push(machine &m, const _value &v)
{
	m.tmp.push_back(v);
}

inline void dump(const machine &m)
{
	std::cout << "=== Machine Dump ===" << std::endl;

	auto q = m.tmp;
	std::cout << "Temporaries:" << std::endl;
	for (auto &v : q)
		std::cout << "\t" << str(v) << std::endl;

	std::cout << "\nStack size: " << m.stack.size() << std::endl;
	for (int i = 0; i < m.stack.size(); i++)
		std::cout << "[" << i << "]: " << str(m.stack[i]) << std::endl;

	std::cout << "\nVariables: " << m.variables.map.size() << std::endl;
	for (auto &p : m.variables.map) {
		std::cout << p.first << " (addr=" << p.second
			<< ", type=" << str(m.variables.types[p.second]) << ")"
			<< " = " << str(m.variables.mem[p.second]) << std::endl;
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
