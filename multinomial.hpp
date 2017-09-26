#ifndef MULTINOMIAL_HPP
#define MULTINOMIAL_HPP

#include "common.hpp"
#include "neuralnet.hpp"

class multinomial: public neuralnet
{
	public:
		int numclasses;
		int c_dim;
		bool lin_an;
		double thresh, var;
		double alpha_thresh_var, lambda_thresh_var;
		vector<int> failed_indices;
		vector<vector<double> > c_theta;
		vector<double> c_bias;
		vector<vector<double> > del_c_theta;
		vector<double> del_c_bias;
		void initialize_predictor(int numc, int cdim);
		vector<double> update_predictor(vector<double> input, vector<double> target, double alph, double mu, double lamb);
		void train_SGD_predictor(vector<vector<double> > samples, vector<vector<double> > targets, double alph, double mu, double lamb,int numiter);
		double test_predictor(vector<vector<double> > samples, vector<vector<double> > targets);
		vector<double> evaluate_prob(vector<double> input);
};

#endif
