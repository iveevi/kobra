#ifndef LEXER_H_
#define LEXER_H_

#define NABU_DEBUG_PARSER
// #define NABU_VERBOSE_OUTPUT

#include "nabu/nabu.hpp"

// TODO: turn into nabu language
inline std::string to_string(const std::string &s) {
	return s;
}

inline std::string strip_quotes(const std::string &s) {
	// case 1: "string"
	// case 2: 'string'
	
	if (s[0] == '"' && s[s.size() - 1] == '"')
		return s.substr(1, s.size() - 2);
	else if (s[0] == '\'' && s[s.size() - 1] == '\'')
		return s.substr(1, s.size() - 2);
	else
		return s; // TODO: throw error
	return s.substr(1, s.size() - 2);
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

struct plus {};
struct minus {};
struct multiply {};
struct divide {};

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

struct lerror {};

ignore(comment)
ignore(space)

// Declaring as tokens
auto_mk_overloaded_token(identifier, "[a-zA-Z_][a-zA-Z0-9_]*", std::string, to_string);
auto_mk_overloaded_token(p_float, "[+-]?[0-9]*\\.[0-9]+", float, std::stof);
auto_mk_overloaded_token(p_int, "[+-]?[0-9]+", int, std::stoi);
auto_mk_overloaded_token(double_str, "\"(?:[^\"\\\\]|\\\\.)*\"", std::string, strip_quotes);
auto_mk_overloaded_token(single_str, "'[^']*'", std::string, strip_quotes);

auto_mk_token(comma, ",");
auto_mk_token(dot, "\\.");
auto_mk_token(equals, "[=]");

auto_mk_token(plus, "[+]");
auto_mk_token(minus, "[-]");
auto_mk_token(multiply, "[*]");
auto_mk_token(divide, "[/]");

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

auto_mk_overloaded_token(lerror, ".", std::string, to_string);

// Define the sequence of tokens to parse
lexlist_next(identifier, p_float);
lexlist_next(p_float, p_int);
lexlist_next(p_int, double_str);

lexlist_next(double_str, single_str);
lexlist_next(single_str, comma);

lexlist_next(comma, dot);
lexlist_next(dot, equals);
lexlist_next(equals, plus);

lexlist_next(plus, minus);
lexlist_next(minus, multiply);
lexlist_next(multiply, divide);
lexlist_next(divide, lparen);

lexlist_next(lparen, rparen);
lexlist_next(rparen, lbracket);

lexlist_next(lbracket, rbracket);
lexlist_next(rbracket, lbrace);

lexlist_next(lbrace, rbrace);
lexlist_next(rbrace, langle);

lexlist_next(langle, rangle);
lexlist_next(rangle, comment);

lexlist_next(comment, space);
lexlist_next(space, lerror);

#endif
