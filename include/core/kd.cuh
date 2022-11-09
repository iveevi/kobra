#ifndef KOBRA_CORE_KD_CUH_
#define KOBRA_CORE_KD_CUH_

namespace kobra {

namespace core {

template <class T>
struct KdNode {
	int		axis;
	float		split;
	float3		point;

	int		parent;
	int		left;
	int		right;

	T		data;
};

}

}

#endif