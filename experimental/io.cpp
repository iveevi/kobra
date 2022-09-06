#include <vector>

#include "value.hpp"

extern "C" {

void print(_value *v, int n)
{
	std::cout << "print-value ptr = " << v << ", n = " << n << std::endl;
	std::cout << "\tvalue: " << str(*v) << std::endl;
}

void import(std::vector <std::string> &signatures)
{
	signatures.push_back("void print(__value__...)");
}

}
