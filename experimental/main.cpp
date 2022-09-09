#include "grammar.hpp"
#include "instruction.hpp"
#include "lexer.hpp"
#include "library.hpp"
#include "nabu/nabu.hpp"

#include <ffi.h>

std::string source = R"(
'21-' + str(12.76 * 6.0/9 + 3.0/5 - (9.0/8 - 19 + 12)) + '-12'
print(13, 14, 'hello world', 12)
if (false)
	int if_1 = 1
else
	int else_1 = 4

float x = 200 * 16 + 10.0/2.5 - 3
int y = 20
string z = "Hello world!"
bool w = false

float t = x * y
)";

machine m;

using nabu::parser::rd::grammar;
using nabu::parser::rd::epsilon;

/*
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
define_action(clause)
{
	push(m, {_instruction::Type::eNcjmp, -1});
	branch_addresses[lptr.get()] = {
		(int) m.instructions.size() - 1, -1
	};
}

define_action(if_branch)
{
	push(m, {_instruction::Type::eJmp, -1});

	_lexvalue *clause = get <vec> (lptr)[1].get();
	_addr_info &info = branch_addresses[clause];
	info.end = m.instructions.size();
}

define_action(else_if_branch)
{
	push(m, {_instruction::Type::eJmp, -1});

	_lexvalue *clause = get <vec> (lptr)[1].get();
	_addr_info &info = branch_addresses[clause];
	info.end = m.instructions.size();
}

using branch = alias <if_branch, repeat <else_if_branch>, option <else_branch, void>>;

define_action(branch)
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

using function_call = alias <
	identifier, lparen, repeat <
		option <alias <expression, comma>, expression>
	>, rparen>; */

struct expression;

struct function_call {
	using production_rule = alias <
		identifier, lparen, repeat <
			option <alias <expression, comma>, expression>
		>, rparen>;
};

struct factor {
	using _parenthesized = alias <lparen, expression, rparen>;
	using production_rule = option <
		function_call,
		_parenthesized,
		primitive,
		variable
	>;
};

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

define_action(term::_mul)
{
	std::cout << "term::_mul" << std::endl;
	push(m, _instruction::Type::eMul);
}

define_action(term::_div)
{
	std::cout << "term::_div" << std::endl;
	push(m, _instruction::Type::eDiv);
}

define_action(expression::_add)
{
	std::cout << "expression::_add" << std::endl;
	push(m, _instruction::Type::eAdd);
}

define_action(expression::_sub)
{
	std::cout << "expression::_sub" << std::endl;
	push(m, _instruction::Type::eSub);
}

define_action(function_call)
{
	std::cout << "function_call" << std::endl;
	vec v = get <vec> (lptr);
	std::string name = get <std::string> (v[0]);
	assert(v.size() == 4);

	v = get <vec> (v[2]);
	int nargs = v.size();
	std::cout << "Nargs = " << nargs << std::endl;
	std::cout << "Elements:" << std::endl;
	for (auto &e : v)
		std::cout << e->str() << std::endl;
	if (m.functions.map_ext.count(name) > 0) {
		// External function
		int index = m.functions.map_ext[name];
		push(m, {_instruction::Type::eCallExt, index, nargs});
		std::cout << "External function" << std::endl;
	} else {
		std::cout << "Unknown function: " << name << std::endl;
	}
}

int main()
{
	using namespace nabu;
	
	// Load libraries
	// TODO: refactor to std
	void *libhandle = dlopen("/home/venki/kobra/bin/lib/libio_arbok.so", RTLD_LAZY);
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
		m.functions.externals.push_back(ext);
		std::cout << "Successfully compiled signature: " << pr.first << std::endl;
	}


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

	using g_input = grammar <expression>;

	parser::rd::DualQueue dq(q);
	g_input::value(dq);

	std::cout << "Top of queue:\n";
	int i = 6;
	while (i--) {
		parser::lexicon lptr = q.front();
		q.pop_front();

		if (lptr == nullptr) {
			std::cout << "nullptr" << std::endl;
			continue;
		}

		std::cout << "\tlexicon: " << lptr->name << " = " << lptr->str() << std::endl;
	}

	// Add an end instruction for padding
	push(m, _instruction::Type::eEnd);

	dump(m);

	exec(m);
	dump(m);

#endif

	return 0;
}
