#include <iostream>
#include <vector>

#include "../../include/arbok/value.hpp"

extern "C" {

void std_print(kobra::arbok::_value *v, int n)
{
	for (int i = 0; i < n; i++)
		std::cout << str(v[i]) << " ";
}

void std_str(std::string *ret, kobra::arbok::_value *v)
{
	*ret = str(*v);
}

void std_len(int *ret, std::string *s)
{
	std::cout << "In std_len" << std::endl;
	std::cout << "len: s = " << *s << std::endl;
	*ret = s->length();
}

void import(std::vector <std::pair <std::string, std::string>> &signatures)
{
	// TODO: some way to check that arguments are correct type
	signatures.push_back({"void print(__value__...)", "std_print"});
	signatures.push_back({"string str(__value__)", "std_str"});
	signatures.push_back({"int len(string)", "std_len"});
}

}
