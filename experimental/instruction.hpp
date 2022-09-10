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

// DLL headers
#include <ffi.h>
#include <dlfcn.h>

// Engine headers
#include "../../include/arbok/value.hpp"

using namespace kobra::arbok;

// Abstraction of the machine state
struct _instruction;

// Function that was imported via dll/ffi
struct _external_function {
	// Signature
	bool variadic;
	int non_variadic_args;
	_value::Type return_type;
	std::string name;
	std::vector <_value::Type> argument_types;

	// Handles
	bool initialized = false;
	void *handle = nullptr;
	ffi_cif cif;
	std::vector <ffi_type *> ffi_arg_types;

	// Default constructor
	_external_function() = default;

	// No copy
	_external_function(const _external_function &) = delete;
	_external_function &operator=(const _external_function &) = delete;

	// Move only
	_external_function(_external_function &&) = default;
	_external_function &operator=(_external_function &&) = default;
};

inline _value::Type get_type(const std::string &str)
{
	bool variadic = false;

	std::string type_str = str;
	if (str.substr(str.size() - 3) == "...") {
		type_str = str.substr(0, str.size() - 3);
		variadic = true;
	}

	int type = -1;
	for (int i = 0; i <= (int) _value::Type::eStruct; i++) {
		if (type_str == _value::type_str[i])
			type = i;
	}

	if (type == -1)
		throw std::runtime_error("Invalid type: " + str);

	return (_value::Type) (type + variadic * _value::Type::eVariadic);
}

inline ffi_type *get_ffi_type(_value::Type type)
{
	switch (type) {
	case _value::Type::eVoid:
		return &ffi_type_void;
	case _value::Type::eGeneric:
		return &ffi_type_pointer;
	case _value::Type::eGeneric + _value::Type::eVariadic:
		return &ffi_type_pointer;
	case _value::Type::eInt:
		return &ffi_type_sint32;
	case _value::Type::eFloat:
		return &ffi_type_float;
	case _value::Type::eBool:
		return &ffi_type_uint8;
	case _value::Type::eString:
		return &ffi_type_pointer;
	default:
		throw std::runtime_error("get_ffi_type: Invalid argument type: " + std::string(_value::type_str[(int) type]));
	}

	return nullptr;
}

inline void push_type(std::vector <ffi_type *> &types, _value::Type type)
{
	switch (type) {
	case _value::Type::eGeneric:
		types.push_back(&ffi_type_pointer);
		break;
	case _value::Type::eGeneric + _value::Type::eVariadic:
		types.push_back(&ffi_type_pointer);
		types.push_back(&ffi_type_sint);
		break;
	default:
		throw std::runtime_error("push_type: Invalid argument type: " + std::string(_value::type_str[(int) type]));
	}
}

inline std::string get_ffi_type_str(ffi_type *type)
{
	if (type == &ffi_type_void)
		return "void";
	if (type == &ffi_type_sint32)
		return "int";
	if (type == &ffi_type_float)
		return "float";
	if (type == &ffi_type_uint8)
		return "bool";
	if (type == &ffi_type_pointer)
		return "pointer";

	return "?";
}

inline _external_function compile_signature(const std::string &sig, const std::string &sym, void *lib)
{
	_external_function ftn;

	// Parse signature
	int space = sig.find(' ');
	int lparen = sig.find('(');
	int rparen = sig.find(')');
	
	std::string ret_type = sig.substr(0, space);
	std::string func_name = sig.substr(space + 1, lparen - space - 1);
	std::string arg_types = sig.substr(lparen + 1, rparen - lparen - 1) + ',';

	std::cout << "ret_type = " << ret_type << std::endl;
	std::cout << "func_name = " << func_name << std::endl;
	std::cout << "arg_types = " << arg_types << std::endl;

	ftn.name = func_name;
	ftn.return_type = get_type(ret_type);
	
	std::string arg_type;

	int nvariadic = 0;
	for (int i = 0; i < arg_types.size(); i++) {
		char c = arg_types[i];
		if (c == ',') {
			std::cout << "Checking arg_type = " << arg_type << std::endl;
			_value::Type type = get_type(arg_type);
			if (type >= _value::Type::eVariadic)
				nvariadic++;

			ftn.argument_types.push_back(type);
			arg_type.clear();
		} else if (!isspace(c)) {
			arg_type += c;
		}
	}

	if (nvariadic > 1) {
		throw std::runtime_error("compile_signature: Can only"
			" have one variadic argument, in \"" + sig + "\""
		);
	}

	ftn.variadic = (nvariadic == 1);
	ftn.non_variadic_args = ftn.argument_types.size() - nvariadic;

	std::string reconstructed_sig = str(ftn.return_type) + " " + func_name + "(";
	for (int i = 0; i < ftn.argument_types.size(); i++) {
		reconstructed_sig += str(ftn.argument_types[i]);
		if (i < ftn.argument_types.size() - 1)
			reconstructed_sig += ", ";
	}
	reconstructed_sig += ")";

	std::cout << "reconstructed_sig = " << reconstructed_sig << std::endl;

	// Load function pointer and ffi
	ftn.handle = dlsym(lib, sym.c_str());
	if (!ftn.handle) {
		fprintf(stderr, "dlsym error: %s\n", dlerror());
		exit(1);
	}
	
	// Initializing arguments and preparing ffi
	std::cout << "Initializing ffi" << std::endl;

	// To be as flexible as possible, there are no
	// explicit return types, only assigning to a pointer
	// as the "return value" (first argument)
	// std::vector <ffi_type *> *ffi_argument_types = new std::vector <ffi_type *>;

	ftn.ffi_arg_types.clear();
	if (ftn.return_type != _value::Type::eVoid)
		ftn.ffi_arg_types.push_back(&ffi_type_pointer);

	for (int i = 0; i < ftn.argument_types.size(); i++)
		push_type(ftn.ffi_arg_types, ftn.argument_types[i]);

	/* std::string signature = ftn.name + "(";
	signature += get_ffi_type_str(get_ffi_type(ftn.return_type));
	if (ftn.argument_types.size() > 0)
		signature += ", ";

	for (int i = 0; i < ffi_argument_types.size(); i++) {
		signature += get_ffi_type_str(ffi_argument_types[i]);
		if (i < ffi_argument_types.size() - 1)
			signature += ", ";
	}

	signature += ")";

	std::cout << "signature = " << signature << std::endl; */

	ffi_status status = ffi_prep_cif(&ftn.cif,
		FFI_DEFAULT_ABI,
		ftn.ffi_arg_types.size(),
		&ffi_type_void,
		ftn.ffi_arg_types.data()
	);

	if (status != FFI_OK) {
		fprintf(stderr, "ffi_prep_cif error: %d\n", status);
		exit(1);
	}

	ftn.initialized = true;
	std::cout << "->Initialized ffi" << std::endl;

	return ftn;
}

inline void *alloc_ret(const _value::Type &type)
{
	switch (type) {
	case _value::Type::eVoid:
		return nullptr;
	case _value::Type::eGeneric:
		return new _value;
	case _value::Type::eInt:
		return new int;
	case _value::Type::eFloat:
		return new float;
	case _value::Type::eBool:
		return new bool;
	case _value::Type::eString:
		return new std::string;
	default:
		throw std::runtime_error("alloc_ret: Invalid return type: " + std::string(_value::type_str[(int) type]));
	}

	return nullptr;
}

inline _value decode_ret(const _value::Type &type, void *ptr)
{
	_value val;
	val.type = type;

	switch (type) {
	case _value::Type::eGeneric:
		val = *(_value *) ptr;
		break;
	case _value::Type::eInt:
		val.data = *(int *) ptr;
		break;
	case _value::Type::eFloat:
		val.data = *(float *) ptr;
		break;
	case _value::Type::eBool:
		val.data = *(bool *) ptr;
		break;
	case _value::Type::eString:
		val.data = *(std::string *) ptr;
		break;
	default:
		throw std::runtime_error("decode_ret: Invalid return type: " + std::string(_value::type_str[(int) type]));
	}

	return val;
}

inline bool type_ok(const _value::Type &type, const _value &val)
{
	switch (type) {
	case _value::Type::eGeneric:
		return true;
	case _value::Type::eInt:
		return val.type == _value::Type::eInt;
	case _value::Type::eFloat:
		return val.type == _value::Type::eFloat;
	case _value::Type::eBool:
		return val.type == _value::Type::eBool;
	case _value::Type::eString:
		return val.type == _value::Type::eString;
	default:
		throw std::runtime_error("type_ok: Invalid argument type: " + std::string(_value::type_str[(int) type]));
	}

	return false;
}

inline _value call(_external_function &ftn, std::vector <_value> &args)
{
	// TODO: # of arguments checking
	std::cout << "# of args passed = " << args.size() << std::endl;
	std::cout << "\t# of non-variadic args = " << ftn.non_variadic_args << std::endl;

	if (ftn.variadic) {
		if (args.size() < ftn.non_variadic_args) {
			throw std::runtime_error("call: Too few arguments passed"
				"to variadic function");
		}
	} else {
		if (args.size() != ftn.non_variadic_args) {
			throw std::runtime_error("call: Incorrect number of"
				"arguments passed to non-variadic function");
		}
	}

	// TODO: type checking
	
	// First check all non-variadic arguments
	for (int i = 0; i < ftn.non_variadic_args; i++) {
		if (!type_ok(ftn.argument_types[i], args[i])) {
			throw std::runtime_error("call: Argument " + std::to_string(i)
				+ " has incorrect type");
		}
	}

	// Then check all variadic arguments
	if (ftn.variadic) {
		_value::Type variadic_type = _value::Type(
			ftn.argument_types[ftn.non_variadic_args]
				- _value::Type::eVariadic
		);

		for (int i = ftn.non_variadic_args; i < args.size(); i++) {
			if (!type_ok(variadic_type, args[i])) {
				throw std::runtime_error("call: Variadic argument "
					+ std::to_string(i - ftn.non_variadic_args)
					+ " has incorrect type");
			}
		}
	}
	
	// Prepare ffi
	// TODO: how to do this only once?
	/* if (!ftn.initialized) {
		std::cout << "Initializing ffi" << std::endl;

		// To be as flexible as possible, there are no
		// explicit return types, only assigning to a pointer
		// as the "return value" (first argument)
		// std::vector <ffi_type *> *ffi_argument_types = new std::vector <ffi_type *>;

		ftn.ffi_arg_types.clear();
		if (ftn.return_type != _value::Type::eVoid)
			ftn.ffi_arg_types.push_back(&ffi_type_pointer);

		for (int i = 0; i < ftn.argument_types.size(); i++)
			push_type(ftn.ffi_arg_types, ftn.argument_types[i]);

		/* std::string signature = ftn.name + "(";
		signature += get_ffi_type_str(get_ffi_type(ftn.return_type));
		if (ftn.argument_types.size() > 0)
			signature += ", ";

		for (int i = 0; i < ffi_argument_types.size(); i++) {
			signature += get_ffi_type_str(ffi_argument_types[i]);
			if (i < ffi_argument_types.size() - 1)
				signature += ", ";
		}

		signature += ")";

		std::cout << "signature = " << signature << std::endl;

		ffi_status status = ffi_prep_cif(&ftn.cif,
			FFI_DEFAULT_ABI,
			ftn.ffi_arg_types.size(),
			&ffi_type_void,
			ftn.ffi_arg_types.data()
		);

		if (status != FFI_OK) {
			fprintf(stderr, "ffi_prep_cif error: %d\n", status);
			exit(1);
		}

		ftn.initialized = true;
		std::cout << "->Initialized ffi" << std::endl;
	} */

	// Prepare arguments
	std::vector <void *> ffi_args;

	// Return value
	if (ftn.return_type != _value::Type::eVoid) {
		void *ret = alloc_ret(ftn.return_type);
		void **ret_ptr = new (void *)(ret);
		ffi_args.push_back(ret_ptr);
	}

	// TODO: custom allocators (goes out of scope after this function)
	for (int i = 0; i < args.size(); i++) {
		_value::Type arg_type = ftn.argument_types[i];
		if (arg_type == _value::Type::eGeneric) {
			_value **ptr = new (_value *)(&args[i]);
			ffi_args.push_back(ptr);
		} else if (arg_type == _value::Type::eGeneric + _value::Type::eVariadic) {
			int *n = new int(args.size() - i);
			_value *ptr = new _value[*n];
			for (int j = i; j < args.size(); j++)
				ptr[j - i] = args[j];

			_value **ptr_ptr = new (_value *)(ptr);
			ffi_args.push_back(ptr_ptr);
			ffi_args.push_back(n);

			// No more arguments
			break;
		} else {
			ffi_args.push_back((void *) &args[i].data);
		}
	}

	// Call function
	ffi_call(&ftn.cif, FFI_FN(ftn.handle), nullptr, ffi_args.data());

	// Decode return value
	if (ftn.return_type != _value::Type::eVoid) {
		void *ret_addr = *(void **) ffi_args[0];
		_value ret = decode_ret(ftn.return_type, ret_addr);
		return ret;
	}

	return _value {_value::Type::eVoid, 0};
}

struct machine {
	// TODO: move to stack frame
	std::vector <_value> stack;
	std::vector <_value> tmp;

	// Functions
	struct {
		std::unordered_map <std::string, int> map_ext;

		std::vector <_external_function> externals;
	} functions;

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

	// Instructions
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
		eCjmp, eNcjmp, eJmp, eCall, eCallExt,
		eRet, eEnd
	} type;

	static constexpr const char *type_str[] = {
		"push_tmp", "push_var", "pop", "store",
		"add", "sub", "mul", "div", "mod",
		"cjmp", "ncjmp", "jmp", "call", "call_ext",
		"ret", "end"
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

		m.stack.push_back(v2 + v1);
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

		m.stack.push_back(v2 * v1);
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

	{_instruction::Type::eCallExt, [](machine &m, const _instruction &i) {
		assert(std::holds_alternative <int> (i.op1));
		assert(std::holds_alternative <int> (i.op2));

		int addr = std::get <int> (i.op1);
		int nargs = std::get <int> (i.op2);

		// Get function
		auto &f = m.functions.externals[addr];

		// Get arguments
		std::vector <_value> args;
		for (int i = 0; i < nargs; i++)
			args.insert(args.begin(), pop(m));

		// Call function
		_value ret = call(f, args);

		// Push return value
		if (ret.type != _value::Type::eVoid)
			m.stack.push_back(ret);
		m.pc++;
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
	std::cout << "\n=== Machine Dump ===" << std::endl;

	auto q = m.tmp;
	std::cout << "Temporaries:" << std::endl;
	for (auto &v : q)
		std::cout << "\t" << info(v) << std::endl;

	std::cout << "\nStack size: " << m.stack.size() << std::endl;
	for (int i = 0; i < m.stack.size(); i++)
		std::cout << "[" << i << "]: " << info(m.stack[i]) << std::endl;

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
