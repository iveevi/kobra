#define NABU_DEBUG_PARSER

#include <nabu/nabu.hpp>

std::string source = R"(
# Defining components:
box = Scene.make_entity('Box')
box.add <Rigidbody> (mass = 200)
box.add <Material> (diffuse = (0.5), roughness = 0.01)

# More components...

# Adding a UI just means assigning to UI.main:
UI.main = final_layout # where final_layout is defined beforehand...
)";

// TODO: turn into nabu language
inline std::string to_string(const std::string &s) {
	return s;
}

// All lexicons
struct identifier {};
struct p_float {};
struct p_int {};
struct double_str {};
struct single_str {};

struct comma {};
struct dot {};
struct equals {};

struct lbracket {};
struct rbracket {};

struct lbrace {};
struct rbrace {};

struct lparen {};
struct rparen {};

struct langle {};
struct rangle {};

struct comment {};
struct space {};

struct error {};

// ignore(comment)
// ignore(space)

// Declaring as tokens
auto_mk_overloaded_token(identifier, "[a-zA-Z_][a-zA-Z0-9_]*", std::string, to_string);
auto_mk_overloaded_token(p_float, "[+-]?[0-9]*\\.[0-9]+", float, std::stof);
auto_mk_overloaded_token(p_int, "[+-]?[0-9]+", int, std::stoi);
auto_mk_overloaded_token(double_str, "\".*\"", std::string, to_string);
auto_mk_overloaded_token(single_str, "'.*'", std::string, to_string);

auto_mk_token(comma, ",");
auto_mk_token(dot, "\\.");
auto_mk_token(equals, "[=]");

auto_mk_token(lbracket, "\\[");
auto_mk_token(rbracket, "\\]");

auto_mk_token(lbrace, "\\{");
auto_mk_token(rbrace, "\\}");

auto_mk_token(lparen, "\\(");
auto_mk_token(rparen, "\\)");

auto_mk_token(langle, "<");
auto_mk_token(rangle, ">");

auto_mk_token(comment, "#[^\n]*");
auto_mk_token(space, "\\s+");

auto_mk_overloaded_token(error, ".", std::string, to_string);

// Define the sequence of tokens to parse
lexlist_next(identifier, p_float);
lexlist_next(p_float, p_int);
lexlist_next(p_int, comma);
lexlist_next(double_str, single_str);

lexlist_next(single_str, comma);
lexlist_next(comma, dot);
lexlist_next(dot, equals);
lexlist_next(equals, lparen);

lexlist_next(lparen, rparen);
lexlist_next(rparen, lbracket);

lexlist_next(lbracket, rbracket);
lexlist_next(rbracket, lbrace);

lexlist_next(lbrace, rbrace);
lexlist_next(rbrace, langle);

lexlist_next(langle, rangle);
lexlist_next(rangle, comment);

lexlist_next(comment, space);
// lexlist_next(space, error);

int main()
{
	using namespace nabu;

	parser::Queue q = parser::lexq <identifier> (source);

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
}
