#include "multinomial.hpp"

void multinomial::initialize_predictor(int numc, int cdim)
{
	numclasses = numc;
	c_dim = cdim;
	thresh = 10.0;
	var = 1.0;
	alpha_thresh_var = 0.01;
	lambda_thresh_var = 0.001;
	c_theta.resize(numclasses);
	c_bias.resize(numclasses,0.0);
	del_c_theta.resize(numclasses);
	del_c_bias.resize(numclasses,0.0);
	for(int i = 0 ; i < numclasses; i++)
	{
		c_theta[i].resize(c_dim,sqrt(6.0)/(c_dim + numclasses));
		del_c_theta[i].resize(c_dim,0.0);
	}

}


vector<double> multinomial::update_predictor(vector<double> input, vector<double> target, double alph, double mu, double lamb)
{
	if(target.size() != numclasses) cerr << "multinomial:: target dimension doesn't match number of classes, target size: " << target.size() << ", numclasses " << numclasses<< endl;
	vector<double> prob(numclasses);
	vector<double> delta_c(numclasses);
	prob = evaluate_prob(input);
	for(int i = 0 ; i< numclasses; i++)
	{
		delta_c[i] = -(target[i] - prob[i]);
		for(int j = 0 ; j < input.size(); j++)
		{
			c_theta[i][j] += -alph*input[j]*delta_c[i] - alph*c_theta[i][j]*lamb + mu*del_c_theta[i][j];
			c_bias[i] += -alph*delta_c[i] + mu*del_c_bias[i];
			del_c_theta[i][j] = -alph*input[j]*delta_c[i] + mu*del_c_theta[i][j];
			del_c_bias[i] = -alph*delta_c[i] + mu*del_c_bias[i];
		}
	}
	vector<double> err(input.size(),0);
	for(int i = 0; i < input.size();i++)
	{
		for(int c = 0 ; c < numclasses; c++) err[i] += delta_c[c]*c_theta[c][i];
	}
	return err;
}

void multinomial::train_SGD_predictor(vector<vector<double> > samples, vector<vector<double> > targets, double alph, double mu, double lamb, int numiter)
{
	vector<double> err(samples[0].size());
	for(int i = 0 ;i < numiter; i++)
	{
		int random_num = gsl_rng_uniform_int(randgen, targets.size());
		vector<double> samples_temp(samples[0].size());
		samples_temp = samples[random_num];
		err = update_predictor(samples_temp, targets[random_num], alph, mu, lamb); 
	}
}

double multinomial::test_predictor(vector<vector<double> > samples, vector<vector<double> > targets)
{
	double numerrors_test = 0;
	failed_indices.resize(0);
	vector<double> prob(numclasses);
	for(int n = 0 ; n < targets.size(); n++)
	{
		prob = evaluate_prob(samples[n]);
		for(int i = 0 ; i < numclasses; i++)
		{
			bool target = targets[n][i] > 0.5 ? 1 : 0 ;
			bool pred = prob[i] > 0.5 ? 1: 0;
			if(target != pred) {numerrors_test += 1.0;failed_indices.push_back(n); break;}
		}
	}
	return numerrors_test/targets.size();
}

vector<double> multinomial::evaluate_prob(vector<double> input)
{
	vector<double> prob(numclasses);
	for(int i = 0; i < numclasses; i++)
	{
		double sum = 0;
		for(int j= 0 ; j < input.size() ; j++ )
		{
			sum+= -c_theta[i][j]*input[j]; 
		}
		prob[i] = 1.0/(1.0 + exp(sum - c_bias[i]));
	}
	return prob;
}

