// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef SUPERVISED_H
#define SUPERVISED_H

#include <ctime>
#include <vector>
#include "../data_structures/matrix.h"
#include "../rand.h"

class SupervisedLearner
{
	public:
		SupervisedLearner() {rng = new Rand(time(NULL));}
		virtual ~SupervisedLearner() {};

		/// Returns the name of this learner
		virtual const char* name() = 0;

		/// Train this learner
		virtual void train(const Matrix& features, const Matrix& labels) = 0;

		/// Forget everything this learner has learned
		virtual void forget() = 0;

		/// Partially train using a single pattern
		virtual void trainIncremental(const Vec& feat, const Vec& lab);

		/// Make a prediction
		virtual const Vec& predict(const Vec& in) = 0;

		/// Measures the misclassifications with the provided test data
		size_t countMisclassifications(const Matrix& features, const Matrix& labels);

		/// Predicts values for the features (calls predict), and compares the answers to the true labels
		double sse(const Matrix& features, const Matrix& labels);

		/// This default implementation just copies the data, without changing it in any way.
		virtual void filter_data(const Matrix& feat_in, const Matrix& lab_in, Matrix& feat_out, Matrix& lab_out);

		void train_with_xvalidation(const Matrix& features, const Matrix& labels, const int repetitions, const int num_folds, const int num_epochs = 1);

	protected:
		Rand* rng;
};


#endif // SUPERVISED_H
