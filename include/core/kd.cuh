#ifndef KOBRA_CORE_KD_CUH_
#define KOBRA_CORE_KD_CUH_

namespace kobra {

namespace core {

struct KdNode {
	int		axis;
	float		split;

	int		left;
	int		right;
};

}

}

#endif