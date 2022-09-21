#ifndef INSTRUCTION_H_
#define INSTRUCTION_H_

// Standard headers
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
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
	bool variadic = false;
	int ivariadic = -1;
	int non_variadic_args;
	Type return_type;
	std::string name;
	std::vector <Type> argument_types;

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

inline Type get_type(const std::string &str, bool &variadic)
{
	variadic = false;

	std::string ltype_str = str;
	if (str.substr(str.size() - 3) == "...") {
		ltype_str = str.substr(0, str.size() - 3);
		variadic = true;
	}

	int type = -1;
	for (int i = 0; i <= (int) Type::eStruct; i++) {
		if (ltype_str == type_str[i])
			type = i;
	}

	if (type == -1)
		throw std::runtime_error("Invalid type: " + str);

	return (Type) (type);
}

inline ffi_type *get_ffi_type(Type type, bool variadic)
{
	switch (type) {
	case Type::eVoid:
		return &ffi_type_void;
	case Type::eGeneric:
		return &ffi_type_pointer;
	case Type::eInt:
		return &ffi_type_sint32;
	case Type::eFloat:
		return &ffi_type_float;
	case Type::eBool:
		return &ffi_type_uint8;
	case Type::eString:
		return &ffi_type_pointer;
	default:
		throw std::runtime_error("get_ffi_type: Invalid argument type: " + std::string(type_str[(int) type]));
	}

	return nullptr;
}

inline void push_type(std::vector <ffi_type *> &types, Type type, bool variadic)
{
	switch (type) {
	case Type::eGeneric:
		if (variadic) {
			types.push_back(&ffi_type_pointer);
			types.push_back(&ffi_type_sint);
		} else {
			types.push_back(&ffi_type_pointer);
		}

		break;
	case Type::eString:
		types.push_back(&ffi_type_pointer);
		break;
	default:
		throw std::runtime_error("push_type: Invalid argument type: " + std::string(type_str[(int) type]));
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

	bool variadic = false;
	ftn.return_type = get_type(ret_type, variadic);
	assert(!variadic);
	
	std::string arg_type;

	int nvariadic = 0;
	for (int i = 0; i < arg_types.size(); i++) {
		char c = arg_types[i];
		if (c == ',') {
			std::cout << "Checking arg_type = " << arg_type << std::endl;
			Type type = get_type(arg_type, variadic);
			if (variadic) {
				nvariadic++;
				ftn.ivariadic = ftn.argument_types.size();
			}

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

	std::string reconstructed_sig = str(ftn.return_type, _value()) + " " + func_name + "(";
	for (int i = 0; i < ftn.argument_types.size(); i++) {
		reconstructed_sig += str(ftn.argument_types[i], _value());
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
	if (ftn.return_type != Type::eVoid)
		ftn.ffi_arg_types.push_back(&ffi_type_pointer);

	for (int i = 0; i < ftn.argument_types.size(); i++) {
		push_type(ftn.ffi_arg_types,
			ftn.argument_types[i],
			i == ftn.ivariadic
		);
	}

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

inline void *alloc_ret(const Type &type)
{
	switch (type) {
	case Type::eVoid:
		return nullptr;
	case Type::eGeneric:
		return new _value;
	case Type::eInt:
		return new int;
	case Type::eFloat:
		return new float;
	case Type::eBool:
		return new bool;
	case Type::eString:
		return new std::string;
	default:
		throw std::runtime_error("alloc_ret: Invalid return type: " + std::string(type_str[(int) type]));
	}

	return nullptr;
}

inline _value decode_ret(const Type &type, void *ptr)
{
	_value val;
	val.type = type;

	switch (type) {
	case Type::eGeneric:
		val = *(_value *) ptr;
		break;
	case Type::eInt:
		val.data = *(int *) ptr;
		break;
	case Type::eFloat:
		val.data = *(float *) ptr;
		break;
	case Type::eBool:
		val.data = *(bool *) ptr;
		break;
	case Type::eString:
		val.data = *(std::string *) ptr;
		break;
	default:
		throw std::runtime_error("decode_ret: Invalid return type: " + std::string(type_str[(int) type]));
	}

	return val;
}

inline bool type_ok(const Type &type, const _value &val)
{
	switch (type) {
	case Type::eGeneric:
		return true;
	case Type::eInt:
		return val.type == Type::eInt;
	case Type::eFloat:
		return val.type == Type::eFloat;
	case Type::eBool:
		return val.type == Type::eBool;
	case Type::eString:
		return val.type == Type::eString;
	default:
		throw std::runtime_error("type_ok: Invalid argument type: " + std::string(type_str[(int) type]));
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
		Type variadic_type = ftn.argument_types
			[ftn.non_variadic_args];

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
		if (ftn.return_type != Type::eVoid)
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
	if (ftn.return_type != Type::eVoid) {
		void *ret = alloc_ret(ftn.return_type);
		void **ret_ptr = new (void *)(ret);
		ffi_args.push_back(ret_ptr);
	}

	// TODO: custom allocators (goes out of scope after this function)
	for (int i = 0; i < args.size(); i++) {
		Type arg_type = ftn.argument_types[i];
		if (i == ftn.ivariadic) {
			// Take care of variadics from here...
			if (arg_type == Type::eGeneric) {
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
				throw std::runtime_error("call: Variadic branch not implemented");
			}
		} else {
			if (arg_type == Type::eGeneric) {
				_value **ptr = new (_value *)(&args[i]);
				ffi_args.push_back(ptr);
			} else {
				void **ptr = new (void *)(&args[i].data);
				ffi_args.push_back(ptr);
			}
		}
	}

	// Call function
	ffi_call(&ftn.cif, FFI_FN(ftn.handle), nullptr, ffi_args.data());

	// Decode return value
	if (ftn.return_type != Type::eVoid) {
		void *ret_addr = *(void **) ffi_args[0];
		_value ret = decode_ret(ftn.return_type, ret_addr);
		return ret;
	}

	return _value {Type::eVoid, 0};
}

struct _type_table {
	std::map <std::string, Type> types;
	std::map <Type, _struct> structs;

	// Type id of the next struct
	int struct_id = Type::eStructId;

	// Default constructor initializes type table
	_type_table() {
		types["void"] = Type::eVoid;
		types["bool"] = Type::eBool;
		types["int"] = Type::eInt;
		types["float"] = Type::eFloat;
		types["string"] = Type::eString;
		types["__value__"] = Type::eGeneric;
	}

	// Add a struct to the type table
	void add_struct(const _struct &s) {
		std::cout << "Adding struct " << s.name << std::endl;

		// Add struct to type table
		types[s.name] = (enum Type) struct_id;
		structs[(enum Type) struct_id] = s;

		// Increment struct id
		struct_id++;
	}
};

struct machine {
	// TODO: move to stack frame
	std::vector <_value> stack;
	std::vector <_value> tmp;

	// Mapped stacks for struct construction
	//	using aggregate initialization
	std::map <std::string, std::stack <_value>>
			member_stacks;

	// Type table
	_type_table type_table;

	// Multiple overloads?

	// Functions
	struct {
		std::unordered_map <std::string, int> map_ext;

		std::vector <_external_function> externals;
	} functions;

	// Stack frame
	struct Frame {
		std::vector <_value> mem;
		std::vector <Type> types;
		std::map <std::string, int> map;

		int add(const std::string &name, Type type) {
			int addr = mem.size();
			map[name] = addr;

			_value v;
			v.type = type;
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
		// Stack operations
		ePushTmp, ePushVar,
		ePushMember, eGetMember,
		ePop, eStore, eConstruct,

		// Arithmetic
		eAdd, eSub, eMul, eDiv, eMod,

		// Comparison
		eCmpEq, eCmpNeq, eCmpLt,
		eCmpGt, eCmpLte, eCmpGte,

		// Program flow operations
		eCjmp, eNcjmp, eJmp,
		eCall, eCallExt,
		eRet, eEnd
	} type;

	static constexpr const char *type_str[] = {
		// Stack operations
		"push_tmp", "push_var",
		"push_member", "get_member",
		"pop", "store", "construct",

		// Arithmetic
		"add", "sub", "mul", "div", "mod",
		// Comparison
		"cmp_eq", "cmp_neq", "cmp_lt",
		"cmp_gt", "cmp_lte", "cmp_gte",

		// Program flow operations
		"cjmp", "ncjmp", "jmp",
		"call", "call_ext",
		"ret", "end"
	};

	// Operands
	using operand_t = std::variant <int, std::string>;

	operand_t op1;
	operand_t op2;

	_instruction(Type t,
			const operand_t &o1 = -1,
			const operand_t &o2 = -1)
			: type(t), op1(o1), op2(o2) {}
};

inline std::string str(const _instruction &i)
{
	std::string out = "(type: ";
	out += _instruction::type_str[(int)i.type];

	out += ", op1: ";
	if (std::holds_alternative <int> (i.op1))
		out += std::to_string(std::get <int> (i.op1));
	else if (std::holds_alternative <std::string> (i.op1))
		out += std::get <std::string> (i.op1);
	else
		throw std::runtime_error("str: Invalid operand type");

	out += ", op2: ";
	if (std::holds_alternative <int> (i.op2))
		out += std::to_string(std::get <int> (i.op2));
	else if (std::holds_alternative <std::string> (i.op2))
		out += std::get <std::string> (i.op2);
	else
		throw std::runtime_error("str: Invalid operand type");

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

	{_instruction::Type::ePushMember, [](machine &m, const _instruction &i) {
		assert(std::holds_alternative <std::string> (i.op1));
		std::string name = std::get <std::string> (i.op1);

		_value v = pop(m);
		m.member_stacks[name].push(v);
		m.pc++;
	}},

	{_instruction::Type::eGetMember, [](machine &m, const _instruction &i) {
		assert(std::holds_alternative <std::string> (i.op1));
		std::string name = std::get <std::string> (i.op1);

		_value v = pop(m);
		assert(v.type > Type::eStruct);
		_struct &s = std::get <_struct> (v.data);
		assert(s.addresses.count(name));
		int saddr = s.addresses[name];
		_value mv = s.members[saddr];

		m.stack.push_back(mv);
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
			auto mv = m.variables.mem[addr];
			std::cerr << "Cannot assign value of type "
				<< str(v.type, v) << " to type "
				<< str(m.variables.types[addr], mv) << std::endl;
			exit(1);
		}

		m.variables.mem[addr] = v;
		m.pc++;
	}},

	{_instruction::Type::eConstruct, [](machine &m, const _instruction &i) {
		// Get the corresponding structs
		Type type = (Type) std::get <int> (i.op1);
		_struct s = m.type_table.structs[type];

		// TODO: make sure everything that is not default is specified

		// Construct the object
		std::string member;

		std::string concatted_members = std::get <std::string> (i.op2);
		for (auto &c : concatted_members) {
			if (c == ';') {
				int saddr = s.addresses[member];

				_value v = m.member_stacks[member].top();
				m.member_stacks[member].pop();

				// Make sure types are matching
				if (s.member_types[saddr] != v.type) {
					std::cerr << "(struct) Cannot assign value of type "
						<< str(v.type, v) << " to type "
						<< str(s.member_types[saddr], m.variables.mem[saddr]) << std::endl;
					exit(1);
				}

				s.members[saddr] = v;

				member.clear();
			} else {
				member += c;
			}
		}

		_value v = {type, s};
		m.stack.push_back(v);

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

	{_instruction::Type::eMod, [](machine &m, const _instruction &i) {
		_value v1 = m.stack.back();
		m.stack.pop_back();

		_value v2 = m.stack.back();
		m.stack.pop_back();

		m.stack.push_back(v2 % v1);
		m.pc++;
	}},

	{_instruction::Type::eCmpEq, [](machine &m, const _instruction &i) {
		_value v1 = m.stack.back();
		m.stack.pop_back();

		_value v2 = m.stack.back();
		m.stack.pop_back();

		m.stack.push_back(v2 == v1);
		m.pc++;
	}},

	{_instruction::Type::eCmpNeq, [](machine &m, const _instruction &i) {
		_value v1 = m.stack.back();
		m.stack.pop_back();

		_value v2 = m.stack.back();
		m.stack.pop_back();

		m.stack.push_back(v2 != v1);
		m.pc++;
	}},

	{_instruction::Type::eCjmp, [](machine &m, const _instruction &i) {
		// Check previous stack value,
		//   then jump if true
		_value v = pop(m);

		if (v.type != Type::eBool) {
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
		if (v.type != Type::eBool) {
			std::cerr << "Conditional clauses must be of type bool" << std::endl;
			std::cerr << "\tncjmp got: " << info(v) << " @(pc = " << m.pc << ")" << std::endl;
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
		if (ret.type != Type::eVoid)
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
		auto mv = m.variables.mem[p.second];
		std::cout << p.first << " (addr=" << p.second
			<< ", type=" << str(m.variables.types[p.second], mv) << ")"
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
