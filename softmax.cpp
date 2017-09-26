#include "softmax.hpp"

void softmax::initialize_predictor(int numc, int cdim)
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


vector<double> softmax::update_predictor(vector<double> input, vector<double> target, double alph, double mu, double lamb)
{
	if(target.size() != numclasses) cerr << "softmax:: target dimension doesn't match number of classes, target size: " << target.size() << ", numclasses " << numclasses<< endl;
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

void softmax::train_SGD_predictor(vector<vector<double> > samples, vector<vector<double> > targets, double alph, double mu, double lamb, int numiter)
{
	vector<double> err(samples[0].size());
	for(int i = 0 ;i < numiter; i++)
	{
		int random_num = gsl_rng_uniform_int(randgen, targets.size());
		vector<double> samples_temp(samples[0].size());
		vector<double> gradient_prefac(samples[0].size(),1.0);
		samples_temp = samples[random_num];
		if(lin_an)
		{
			for (int j= 0 ; j < samples_temp.size(); j++)
			{
				//samples_temp[j] = 1.0/(1.0 + exp(-(samples[random_num][j]- thresh)/var));
				samples_temp[j] = samples[random_num][j]*var - thresh;
				if(samples_temp[j] < 0)
				{
					samples_temp[j] = 0.01*samples_temp[j];
					gradient_prefac[j] = 0.01;
				}
			}
		}
		double grad_thresh = 0;
		double grad_var = 0;
		err = update_predictor(samples_temp, targets[random_num], alph, mu, lamb); 
			
		if(lin_an)
		{
			for(int j = 0 ; j < err.size(); j++)
			{
				//grad_thresh += err[j]*samples_temp[j]*(1.0 - samples_temp[j])/var;
				//grad_var += (thresh/var)*err[j]*samples_temp[j]*(1.0 - samples_temp[j])/var;
				grad_thresh += -gradient_prefac[j]*err[j];
				grad_var += gradient_prefac[j]*err[j]*samples[random_num][j];
			}
			thresh += -alpha_thresh_var*grad_thresh/err.size() - lambda_thresh_var*thresh/err.size();
			var += -alpha_thresh_var*grad_var/err.size() - lambda_thresh_var*var/err.size();
		}
		//cout << thresh << " " << grad_thresh << " " << var << " " << grad_var << endl;
	}

}

double softmax::test_predictor(vector<vector<double> > samples, vector<vector<double> > targets)
{
	double numerrors_test = 0;
	failed_indices.resize(0);
	vector<double> prob(numclasses);
	for(int n = 0 ; n < targets.size(); n++)
	{
		prob = evaluate_prob(samples[n]);
		double maxprob = -1; int argmaxprob = 0; int trueclass =0 ;
		for(int i = 0 ; i< numclasses; i++)
		{
			if(maxprob < prob[i]){maxprob = prob[i]; argmaxprob = i;}
			if(targets[n][i] > 0.5) trueclass = i;
		}
		if(trueclass != argmaxprob) {numerrors_test += 1.0; failed_indices.push_back(n);}
	}
	return numerrors_test/targets.size();
}

vector<double> softmax::evaluate_prob(vector<double> sample)
{
	vector<double> prob(numclasses);
	double normalizer = 0;
	for(int i = 0; i < numclasses; i++)
	{
		double sum = 0;
		for(int j= 0 ; j < sample.size() ; j++ )
		{
			sum+= c_theta[i][j]*sample[j]; 
		}
		prob[i] = exp(sum + c_bias[i]);
		normalizer += prob[i];
	}
	for(int i = 0 ; i < numclasses; i++)
	{
		prob[i] /= normalizer;
	}
	return prob;
}

