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

Point { x: 165, y: 243 - 40 + 3 }
Point pt = Point { x: 1, y: 2 }

if (len('hi') == 2)
	int if_1 = 1
else
	int else_1 = 4

float x = 200 * 16 + 10.0/2.5 - 3
int y = 20
string z = "Hello world!"
bool w = false
float t = x * y
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
	push(m, {_instruction::Type::eNcjmp, -1});
	branch_addresses[lptr.get()] = {
		(int) m.instructions.size() - 1, -1
	};
}

nabu_define_action(if_branch)
{
	push(m, {_instruction::Type::eJmp, -1});

	_lexvalue *clause = get <vec> (lptr)[1].get();
	_addr_info &info = branch_addresses[clause];
	info.end = m.instructions.size();
}

nabu_define_action(else_if_branch)
{
	push(m, {_instruction::Type::eJmp, -1});

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

// Struct construction
// 	All members should be initialized,
// 	unless a default value has been specified
// 	in the definition of the struct.
struct p_struct {
	using member_init = alias <identifier, colon, expression>;
	using member_init_list = alias <
		member_init,
		repeat <alias <comma, member_init>>
	>;

	using production_rule = alias <
		type,
		lbrace, member_init_list, rbrace
	>;
};

nabu_define_action(p_struct::member_init)
{
	std::cout << "p_struct::member_init" << std::endl;
	std::cout << lptr->str() << std::endl;

	vec v = get <vec> (lptr);
	std::string member = get <std::string> (v[0]);

	push(m, {_instruction::Type::ePushMember, member});
}

nabu_define_action(p_struct)
{
	std::cout << "p_struct" << std::endl;
	std::cout << lptr->str() << std::endl;

	// Get type
	vec v = get <vec> (lptr);

	Type type = get <Type> (v[0]);
	std::cout << "Type: " << (int) type << std::endl;

	// Get all the members that were explicitly initialized
	vec member_inits = get <vec> (v[2]);

	std::vector <std::string> explicit_members;
	
	// First element is guaranteed to be a member_init
	vec m1 = get <vec> (member_inits[0]);
	explicit_members.push_back(get <std::string> (m1[0]));

	// The rest are optional, iterate over them...
	vec rest = get <vec> (member_inits[1]);
	for (auto &r : rest) {
		std::cout << "rest = " << r->str() << std::endl;
		vec v = get <vec> (r);
		std::cout << "\tv[1] = " << v[1]->str() << std::endl;
		vec m = get <vec> (v[1]);
		explicit_members.push_back(get <std::string> (m[0]));
	}

	std::cout << "Explicit members: " << std::endl;
	for (auto &m : explicit_members)
		std::cout << "\t" << m << std::endl;

	std::string concatted_members;
	for (auto &m : explicit_members)
		concatted_members += m + ";";

	// Get corresponding struct
	assert(m.type_table.structs.count(type));
	_struct s = m.type_table.structs[type];

	std::cout << "struct: " << str(s, true) << std::endl;

	push(m, {_instruction::Type::eConstruct, (int) type, concatted_members});
}

struct statement : public option <s_struct, assignment> {};

int main()
{
	using namespace nabu;

	// Import standard library
	import("/home/venki/kobra/bin/lib/libarbok.so", m);

	// Read lexicons
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

		std::cout << "lexicon: " << lptr->name << " = " << lptr->str() << std::endl;
	}

#else

	using g_input = grammar <s_struct>;

	parser::rd::DualQueue dq(q);

	grammar <s_struct> ::value(dq);
	grammar <p_struct> ::value(dq);

	// while(g_input::value(dq));

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
	push(m, _instruction::Type::eEnd);

	dump(m);

	exec(m);
	dump(m);

#endif

	return 0;
}
