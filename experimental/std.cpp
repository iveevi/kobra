#include <iostream>
#include <vector>

#include "../../include/arbok/value.hpp"

extern "C" {

void std_print(kobra::arbok::_value *v, int n)
{
	std::cout << "print-n = " << n << std::endl;
	for (int i = 0; i < n; ++i)
		std::cout << "\t" << str(v[i]) << std::endl;
}

void std_str(std::string *ret, kobra::arbok::_value *v)
{
	*ret = str(*v);
}

void import(std::vector <std::pair <std::string, std::string>> &signatures)
{
	// TODO: some way to check that arguments are correct type
	signatures.push_back({"void print(__value__...)", "std_print"});
	signatures.push_back({"string str(__value__)", "std_str"});
}

}
