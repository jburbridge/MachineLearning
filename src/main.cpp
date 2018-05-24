// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <exception>
#include <string>
#include <memory>
#include "error.h"
#include "image.h"
#include "rand.h"
#include "string.h"
#include "svg.h"
#include "data_structures/matrix.h"
#include "data_structures/tensor3d.h"
#include "learners/supervised.h"
#include "learners/baseline.h"
#include "layers/layerconv.h"
#include "layers/layerlinear.h"
#include "layers/layerlrelu.h"
#include "layers/layermaxpooling2d.h"
#include "layers/layersin.h"
#include "layers/layertanh.h"
#include "learners/neuralnet.h"
#include "preprocessing/nomcat.h"

using std::cout;
using std::cerr;
using std::string;
using std::auto_ptr;

int main(int argc, char *argv[])
{
	//std::clock_t start = std::clock();
	//srand(time(NULL));
	enableFloatingPointExceptions();
	int ret = 1;

	try
	{
		cout << "Hello!\n";
		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	cout.flush();
	cerr.flush();


	return ret;
}
