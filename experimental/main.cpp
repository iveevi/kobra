#include "grammar.hpp"
#include "instruction.hpp"
#include "lexer.hpp"
#include "library.hpp"
#include "nabu/nabu.hpp"

#include <ffi.h>

std::string source = R"(
struct Point {
    x: int,
    y: int
}

int x = 1

func myfunc(a: int, b: int) -> int {
    # return a + b
}

func myprint(a: int) {
    print(a)
}
)";

// Point pt = Point { x: 1, y: 2 }
// int x = pt.x

std::string source2 = R"(
if (len('hi') == 3)
	int if_1 = 1
else
	int else_1 = 4

int x = 12
int y = 20
string z = "Hello world!"
bool w = false
float t = 2.5 * x * y
print('t =' , t, 'z = \'', z, '\'')
)";

machine m;

using nabu::parser::rd::grammar;
using nabu::parser::_lexvalue;

using clause = alias <lparen, expression, rparen>;
using body = alias <lbrace, repeat <statement>, rbrace>;
using conditional_body = option <statement, body>;

using if_branch = alias <k_if, clause, conditional_body>;
using else_branch = alias <k_else, conditional_body>;

using k_else_if = alias <k_else, k_if>;
using else_if_branch = alias <k_else_if, clause, conditional_body>;

register(k_else_if)
register(if_branch)
register(else_branch)
register(else_if_branch)

// TODO: custom allocator for improved performance?

struct _addr_info {
	int ncjmp;
	int end;
};

std::map <_lexvalue *, _addr_info> branch_addresses;

// Clauses are for braching and loops:
// 	evaluate the expression in the clause and
// 	only then possibly jump
nabu_define_action(clause)
{
	push_instr(m, {_instruction::Type::eNcjmp, -1});
	branch_addresses[lptr.get()] = {
		(int) m.instructions.size() - 1, -1
	};
}

nabu_define_action(if_branch)
{
	push_instr(m, {_instruction::Type::eJmp, -1});

	_lexvalue *clause = get <vec> (lptr)[1].get();
	_addr_info &info = branch_addresses[clause];
	info.end = m.instructions.size();
}

nabu_define_action(else_if_branch)
{
	push_instr(m, {_instruction::Type::eJmp, -1});

	_lexvalue *clause = get <vec> (lptr)[1].get();
	_addr_info &info = branch_addresses[clause];
	info.end = m.instructions.size();
}

using branch = alias <if_branch, repeat <else_if_branch>, option <else_branch, epsilon>>;

nabu_define_action(branch)
{
	// Resolve jump addresses
	std::cout << "lptr = " << lptr->str() << std::endl;
	vec v = get <vec> (lptr);

	// Should always be 3 elements, even without else-if and else
	assert(v.size() == 3);

	// Get the clauses in the branch address map
	std::vector <_lexvalue *> clauses;

	// If-branch
	_lexvalue *if_clause = get <vec> (v[0])[1].get();
	clauses.push_back(if_clause);

	// Else-if-branches
	vec else_ifs = get <vec> (v[1]);
	for (auto &else_if : else_ifs) {
		_lexvalue *else_if_clause = get <vec> (else_if)[1].get();
		clauses.push_back(else_if_clause);
	}

	// We don't need the else-branch clause,
	// 	since it's always the last one
	std::cout << "Clauses" << std::endl;
	for (auto &clause : clauses)
		std::cout << clause << std::endl;


	std::cout << "Branch addresses" << std::endl;
	for (auto &p : branch_addresses)
		std::cout << p.first << " -> (" << p.second.ncjmp << ", " << p.second.end << ")" << std::endl;

	// Fix negative conditional jump (eNcjmp) addresses
	for (int i = 0; i < clauses.size(); i++) {
		_addr_info &info = branch_addresses[clauses[i]];
		int pc = info.ncjmp;
		int address = info.end;

		_instruction &instruction = m.instructions[pc];
		assert(instruction.type == _instruction::Type::eNcjmp);
		instruction.op1 = address;
	}
	
	// Fix jump (eJmp) addresses after body of each branch
	int end = m.instructions.size();
	for (int i = 0; i < clauses.size(); i++) {
		_addr_info &info = branch_addresses[clauses[i]];
		_instruction &instruction = m.instructions[info.end - 1];
		assert(instruction.type == _instruction::Type::eJmp);
		instruction.op1 = end;
	}
}

void import(const std::string &path, machine &m)
{
	// Load libraries
	void *libhandle = dlopen(path.c_str(), RTLD_LAZY);
	if (!libhandle) {
		fprintf(stderr, "dlopen error: %s\n", dlerror());
		exit(1);
	}

	printf("dlopen success: handle %p\n", libhandle);

	typedef void (*importer_t)(std::vector <std::pair <std::string, std::string>> &);
	importer_t func = (importer_t) dlsym(libhandle, "import");
	if (!func) {
		fprintf(stderr, "dlsym error: %s\n", dlerror());
		exit(1);
	}

	printf("dlsym success: func %p\n", func);

	std::vector <std::pair <std::string, std::string>> args;
	func(args);

	for (auto pr : args) {
		auto ext = compile_signature(pr.first, pr.second, libhandle);
		m.functions.map_ext.insert({ext.name, m.functions.externals.size()});
		m.functions.externals.emplace_back(std::move(ext));
		std::cout << "Successfully compiled signature: " << pr.first << std::endl;
	}

}

struct s_struct {
	// TODO: type should be an identifier, just aliased... (like
	// variable, then check if type is valid)
	using member = alias <identifier, colon, type>;
	using member_list = alias <member, repeat <alias <comma, member>>>;

	using production_rule = alias <
		k_struct, identifier,
		lbrace, member_list, rbrace
	>;
};

nabu_define_action(s_struct)
{
	// TODO: make sure member names are unique
	vec v = get <vec> (lptr);
	assert(v.size() == 5);

	// Get the struct name
	std::string name = get <std::string> (v[1]);
	std::cout << "Struct name: " << name << std::endl;

	// Get the member list
	vec members = get <vec> (v[3]);

	vec member_list;
	member_list.push_back(members[0]);

	vec rest = get <vec> (members[1]);
	for (auto &r : rest) {
		vec v = get <vec> (r);
		member_list.push_back(v[1]);
	}

	std::vector <std::string> member_names;
	std::vector <Type> member_types;
	for (auto &m : member_list) {
		std::cout << "\tm = " << m->str() << std::endl;
		vec v = get <vec> (m);
		std::string member_name = get <std::string> (v[0]);
		Type member_type = get <Type> (v[2]);

		member_names.push_back(member_name);
		member_types.push_back(member_type);
	}

	_struct s = make_struct(name, member_names, member_types);
	m.type_table.add_struct(s);
}

nabu_define_action(p_struct::member_init)
{
	std::cout << "p_struct::member_init" << std::endl;
	std::cout << lptr->str() << std::endl;

	vec v = get <vec> (lptr);
	std::string member = get <std::string> (v[0]);

	push_instr(m, {_instruction::Type::ePushMember, member});
}

nabu_define_action(p_struct)
{
	// Get type
	vec v = get <vec> (lptr);

	Type type = get <Type> (v[0]);

	// Get all the members that were explicitly initialized
	vec member_inits = get <vec> (v[2]);

	std::vector <std::string> explicit_members;
	
	// First element is guaranteed to be a member_init
	vec m1 = get <vec> (member_inits[0]);
	explicit_members.push_back(get <std::string> (m1[0]));

	// The rest are optional, iterate over them...
	vec rest = get <vec> (member_inits[1]);
	for (auto &r : rest) {
		vec v = get <vec> (r);
		vec m = get <vec> (v[1]);
		explicit_members.push_back(get <std::string> (m[0]));
	}

	std::string concatted_members;
	for (auto &m : explicit_members)
		concatted_members += m + ";";

	// Get corresponding struct
	assert(m.type_table.structs.count(type));
	_struct s = m.type_table.structs[type];

	push_instr(m, {_instruction::Type::eConstruct, (int) type, concatted_members});
}

struct statement : public option <
	s_struct,
	assignment,
	branch,
	expression
> {};

struct s_function {
	using parameter = alias <identifier, colon, type>;
	using parameter_list = alias <parameter, repeat <alias <comma, parameter>>>;
	using signature = alias <
		k_func, identifier,
		lparen, option <parameter_list, epsilon>, rparen
	>;

	using return_type = alias <sym_return, type>;
	using function_end = alias <rbrace>;
	
	using production_rule = alias <
		signature,
		option <return_type, epsilon>,
		lbrace, repeat <statement>, function_end
	>;
};

nabu_define_action(s_function::signature)
{
	std::cout << "s_function::signature" << std::endl;
	std::cout << lptr->str() << std::endl;

	// Gather all the arguments
	vec v = get <vec> (lptr);
	vec args = get <vec> (v[3]);

	std::vector <std::pair <std::string, Type>> arg_list;

	vec a1 = get <vec> (args[0]);
	arg_list.push_back({get <std::string> (a1[0]), get <Type> (a1[2])});

	args = get <vec> (args[1]);
	for (auto &a : args) {
		vec v = get <vec> (a);
		vec a2 = get <vec> (v[1]);
		arg_list.push_back({get <std::string> (a2[0]), get <Type> (a2[2])});
	}

	std::cout << "Args: " << std::endl;
	for (auto &a : arg_list)
		std::cout << "\t" << a.first << " : " << a.second << std::endl;

	// Push jump to end of function,
	//	so that we dont execute
	//	the function body accidentally
	// push(m, {_instruction::Type::eJmp, -1});

	// Then push a frame
	m.frames.push_back(machine::Frame {});

	// # of args is passed to allocate the space in the frame
	push_instr(m, {_instruction::Type::ePushFrame, (int) arg_list.size()});

	// Push variables onto frame
	for (auto &a : arg_list)
		m.frames.back().add(a.first, a.second);

	// Then store top stack elements:
	// 	should always be the first n elements,
	// 	where n is the number of arguments
	int level = m.frames.size() - 1;
	for (int i = 0; i < arg_list.size(); i++)
		push_instr(m, {_instruction::Type::eStore, level, i});
}

nabu_define_action(s_function::function_end)
{
	std::cout << "s_function::function_end" << std::endl;

	// Pop the frame
	m.frames.pop_back();
	push_instr(m, {_instruction::Type::ePopFrame});
}

int main()
{
	using namespace nabu;

	// Import standard library
	import("/home/venki/kobra/bin/lib/libarbok.so", m);

	// Read lexicons
	parser::Queue q = parser::lexq <identifier> (source2);

#if 0

	std::cout << "Queue size: " << q.size() << std::endl;
	while (!q.empty()) {
		parser::lexicon lptr = q.front();
		q.pop_front();

		if (lptr == nullptr) {
			std::cout << "nullptr" << std::endl;
			continue;
		}

		std::cout << "lexicon: " << lptr->name << " = " << lptr->str() << std::endl;
	}

#else

	using g_input = grammar <statement>;

	parser::rd::DualQueue dq(q);

	while(g_input::value(dq));
	grammar <s_function>::value(dq);

	/* std::cout << "Top of queue:\n";
	int i = 6;
	while (i--) {
		parser::lexicon lptr = q.front();
		q.pop_front();

		if (lptr == nullptr) {
			std::cout << "nullptr" << std::endl;
			continue;
		}

		std::cout << "\tlexicon: " << lptr->str() << std::endl;
	} */

	// Add an end instruction for padding
	push_instr(m, _instruction::Type::eEnd);

	dump(m);

	exec(m);
	dump(m);

#endif

	return 0;
}
