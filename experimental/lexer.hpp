#ifndef LEXER_H_
#define LEXER_H_

#define NABU_DEBUG_PARSER
// #define NABU_VERBOSE_OUTPUT

#include "nabu/nabu.hpp"

// Engine headers
#include "value.hpp"

// TODO: turn into nabu language
inline std::string to_string(const std::string &s) {
	return s;
}

inline std::string convert_string(_value::Type e)
{
	if (e > _value::Type::eStruct)
		return "?";

	return _value::type_str[(int) e];
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

inline _value::Type to_type(const std::string &s)
{
	for (int i = 0; i <= (int)_value::Type::eStruct; i++)
		if (s == _value::type_str[i])
			return (_value::Type) i;

	return _value::Type::eVoid;
}

//////////////////
// All lexicons //
//////////////////

// Reserved words
nabu_terminal(type);

nabu_terminal(k_if);
nabu_terminal(k_else);
nabu_terminal(k_while);

nabu_terminal(k_true);
nabu_terminal(k_false);

nabu_terminal(identifier);
nabu_terminal(p_float);

nabu_terminal(p_int);

nabu_terminal(double_str);
nabu_terminal(single_str);

nabu_terminal(comma);
nabu_terminal(dot);
nabu_terminal(equals);

nabu_terminal(plus);
nabu_terminal(minus);
nabu_terminal(multiply);
nabu_terminal(divide);

nabu_terminal(lbracket);
nabu_terminal(rbracket);

nabu_terminal(lbrace);
nabu_terminal(rbrace);

nabu_terminal(lparen);
nabu_terminal(rparen);

nabu_terminal(langle);
nabu_terminal(rangle);

nabu_terminal(comment);
nabu_terminal(space);

nabu_terminal(lerror);

ignore(comment)
ignore(space)

/////////////////////////////////////
// Defining tokens (lexicon regex) //
/////////////////////////////////////

// All reserved words
#define RESERVED_WORDS "if|else|while|true|false|int|float|string|bool"

auto_mk_overloaded_token(identifier, "\\b(?!(?:" RESERVED_WORDS ")\\b)[a-zA-Z_][a-zA-Z0-9_]+\\b", std::string, to_string)

// Reserved words
auto_mk_overloaded_token(type, "int|float|string|bool", _value::Type, to_type)

auto_mk_token(k_if, "if")
auto_mk_token(k_else, "else")
auto_mk_token(k_while, "while")

auto_mk_overloaded_token(k_true, "true", bool, [](const std::string &s) { return true; })
auto_mk_overloaded_token(k_false, "false", bool, [](const std::string &s) { return false; })

auto_mk_overloaded_token(p_float, "[+-]?[0-9]*\\.[0-9]+", float, std::stof)
auto_mk_overloaded_token(p_int, "[+-]?[0-9]+", int, std::stoi)
auto_mk_overloaded_token(double_str, "\"(?:[^\"\\\\]|\\\\.)*\"", std::string, strip_quotes)
auto_mk_overloaded_token(single_str, "'[^']*'", std::string, strip_quotes)

auto_mk_token(comma, ",")
auto_mk_token(dot, "\\.")
auto_mk_token(equals, "[=]")

auto_mk_token(plus, "[+]")
auto_mk_token(minus, "[-]")
auto_mk_token(multiply, "[*]")
auto_mk_token(divide, "[/]")

auto_mk_token(lbracket, "\\[")
auto_mk_token(rbracket, "\\]")

auto_mk_token(lbrace, "\\{")
auto_mk_token(rbrace, "\\}")

auto_mk_token(lparen, "\\(")
auto_mk_token(rparen, "\\)")

auto_mk_token(langle, "<")
auto_mk_token(rangle, ">")

auto_mk_token(comment, "#[^\n]*")
auto_mk_token(space, "\\s+")

auto_mk_overloaded_token(lerror, ".", std::string, to_string)

// Define the sequence of tokens to parse
lexlist_next(identifier, type)
lexlist_next(type, k_if)

lexlist_next(k_if, k_else)
lexlist_next(k_else, k_while)
lexlist_next(k_while, k_true)

lexlist_next(k_true, k_false)
// lexlist_next(k_false, identifier)
lexlist_next(k_false, p_float)

// lexlist_next(identifier, p_float)
lexlist_next(p_float, p_int)
lexlist_next(p_int, double_str)

lexlist_next(double_str, single_str)
lexlist_next(single_str, comma)

lexlist_next(comma, dot)
lexlist_next(dot, equals)
lexlist_next(equals, plus)

lexlist_next(plus, minus)
lexlist_next(minus, multiply)
lexlist_next(multiply, divide)
lexlist_next(divide, lparen)

lexlist_next(lparen, rparen)
lexlist_next(rparen, lbracket)

lexlist_next(lbracket, rbracket)
lexlist_next(rbracket, lbrace)

lexlist_next(lbrace, rbrace)
lexlist_next(rbrace, langle)

lexlist_next(langle, rangle)
lexlist_next(rangle, comment)

lexlist_next(comment, space)
lexlist_next(space, lerror)

#endif
