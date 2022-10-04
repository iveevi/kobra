#ifndef KOBRA_OPTIX_OPTIONS_H_
#define KOBRA_OPTIX_OPTIONS_H_

namespace kobra {

namespace optix {

enum SamplingStrategies : unsigned int {
	eDefault,
	eTemporal,
	eSpatioTemporal,
	eMax
};

}

}

#endif
