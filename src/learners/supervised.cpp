#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include "supervised.h"
#include "../error.h"
#include "../data_structures/string.h"
#include "../rand.h"
#include "../data_structures/vec.h"

using std::vector;

// virtual
void SupervisedLearner::filter_data(const Matrix& feat_in, const Matrix& lab_in, Matrix& feat_out, Matrix& lab_out)
{
	feat_out.copy(feat_in);
	lab_out.copy(lab_in);
}

size_t SupervisedLearner::countMisclassifications(const Matrix& features, const Matrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Mismatching number of rows");

	size_t mis = 0;

	for(size_t i = 0; i < features.rows(); i++)
	{
		const Vec& pred = predict(features[i]);
		const Vec& lab = labels[i];

		if(pred.indexOfMax() != lab.indexOfMax())
			mis++;
	}

	return mis;
}

double SupervisedLearner::sse(const Matrix& features, const Matrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Mismatching number of rows");

	double error = 0.0;

	for(size_t i = 0; i < features.rows(); i++)
	{
		const Vec& pred = predict(features[i]);
		const Vec& lab = labels[i];

		for(size_t j = 0; j < pred.size(); j++)
			std::cout << pred[j] << std::endl;

		for(size_t j = 0; j < lab.size(); j++)
			error += (pred[j] - lab[j]) * (pred[j] - lab[j]);
	}
	return error;
}

// virtual
void SupervisedLearner::trainIncremental(const Vec& feat, const Vec& lab)
{
	throw Ex("Sorry, this learner does not support incremental training");
}

void SupervisedLearner::train_with_xvalidation(const Matrix& features, const Matrix& labels, const int repetitions, const int num_folds, const int num_epochs)
{
	for(int i = 0; i < repetitions; i++)
	{
		//Make an array specifying order
		size_t numSwaps = features.rows();
		int *order = new int[features.rows()];
		for(size_t i = 0; i < features.rows(); i++)
			order[i] = (int)i;

		//Do a bunch of random swaps
		int s1, s2, temp;
		for(size_t s = 0; s < numSwaps; s++)
		{
			s1 = rng->next() % features.rows();
			s2 = rng->next() % features.rows();
			temp = order[s1];
			order[s1] = order[s2];
			order[s2] = temp;
		}

		//Use order to create folds
		Matrix shuffled_feat(features.rows(), features.cols());
		Matrix shuffled_lab(labels.rows(), labels.cols());
		for(size_t j = 0; j < features.rows(); j++)
		{
			shuffled_feat.copyBlock(j, 0, features, order[j], 0, 1, features.cols());
			shuffled_lab.copyBlock(j, 0, labels, order[j], 0, 1, labels.cols());
		}

		size_t foldSize = shuffled_feat.rows() / num_folds;

		//Create train and test sets
		Matrix train_lab(shuffled_feat.rows() - foldSize, shuffled_lab.cols());
		Matrix train_feat(shuffled_feat.rows() - foldSize, shuffled_feat.cols());
		Matrix test_lab(foldSize, shuffled_lab.cols());
		Matrix test_feat(foldSize, shuffled_feat.cols());
		for(size_t j = 0; j < (size_t)num_folds; j++)
		{
			size_t testStart = j*foldSize;
			size_t testEnd = j*foldSize+foldSize;
			if(testEnd > shuffled_feat.rows())
				testEnd = shuffled_feat.rows();

			train_feat.copyBlock(0, 0, shuffled_feat, 0,  0, testStart, shuffled_feat.cols());
			train_lab.copyBlock(0, 0, shuffled_lab, 0,  0, testStart, shuffled_lab.cols());

			test_feat.copyBlock(0, 0, shuffled_feat, testStart,  0, foldSize, shuffled_feat.cols());
			test_lab.copyBlock(0, 0, shuffled_lab, testStart,  0, foldSize, shuffled_lab.cols());

			train_feat.copyBlock(testStart, 0, shuffled_feat, testEnd,  0, shuffled_feat.rows() - testEnd, shuffled_feat.cols());
			train_lab.copyBlock(testStart, 0, shuffled_lab, testEnd,  0, shuffled_lab.rows() - testEnd, shuffled_lab.cols());

			//Train
			for(int e = 0; e < num_epochs; e++)
				train(train_feat, train_lab);
			std::cout << "Rep " << i+1  << "/" << repetitions << ", Fold " << j+1 << "/" << num_folds << ": "
					  << countMisclassifications(test_feat, test_lab) << "\n";

			//Reset learner between folds
			forget();
		}
	}
}
