#include "multinomial.hpp"
#include "softmax.hpp"


int N = 250;

int numtarget = 1;
int numclasses = numtarget;
int numbgs = 5;
int numtrain = 1000;
int numtest = 5000;
int max_numt = 1;

double alpha = 0.005;
double mu = 0.5;
double lamb = 1e-2;

double rho = 0.0;
double sparsity = 0.5;
int hillc = 4;
double thresh = 0.5;

double thresh_classifier = 0.5;

double lnc_t = 10*log(10);
double c_ratio = 0.0*log(10);
vector<double> prob_nt;

int mode = 0; // mode = 0 is infinite conc., in mode 1 you can vary it. 

int draw_numb()
{
	int temp = min(1 + (int) (-numtarget*log(gsl_rng_uniform_pos(randgen))/4.0), numtarget-1);
	return temp;
	//return floor(exp(log(1.0*numb)*gsl_rng_uniform_pos(randgen)));
}


template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  return idx;
}

vector<double> generate_sample(int chosenclass, int currtarget, const vector<vector<double> > & kappa_t,const vector<vector<double> > & eta_t)
{
	vector<double> cov(N,0.0);
	int nb_temp = chosenclass > 0.5 ? 0 : 1 ;
	nb_temp += (int) ceil((max_numt-1)*gsl_rng_uniform_pos(randgen));

	double targetbag[numtarget-1];
	int flag = 0;
	for(int i =0 ; i < numtarget; i++) 
	{
		if(i == currtarget){flag = 1;continue;}
		targetbag[i-flag] = (double) i ; 
	}
	double chosen[nb_temp];
	gsl_ran_choose(randgen, chosen, nb_temp, targetbag, numtarget - 1 , sizeof(double));
	gsl_ran_shuffle(randgen, chosen, nb_temp, sizeof(double));

	
	vector<double> cb(nb_temp);
	//double ct = -exp(lnc_t)*log(gsl_rng_uniform_pos(randgen)); 
	double ct = exp(log(10)*lnc_t*(gsl_rng_uniform_pos(randgen) - 0.5) + 5*log(10));
	
	if(mode == 1)
	{
		for(int l = 0 ; l < nb_temp; l++)
		{
			//cb[l] = -exp(lnc_b)*log(gsl_rng_uniform_pos(randgen)); 
			cb[l] = exp(log(10)*lnc_t*(gsl_rng_uniform_pos(randgen) - 0.5) + 5*log(10));
		}
	}

	for(int i = 0 ; i < N; i++)
	{
		if(mode == 0)
		{
			double eta_sum = 0;
			double kappa_sum = 0;
			if (chosenclass > 0.5) 
			{
				eta_sum += eta_t[currtarget][i]/kappa_t[currtarget][i];
				kappa_sum += 1.0/kappa_t[currtarget][i];
			}
			for(int j = 0 ;j < nb_temp; j++) 
			{
				eta_sum += eta_t[chosen[j]][i]/kappa_t[chosen[j]][i];
				kappa_sum += 1.0/kappa_t[chosen[j]][i];
			}
			double eta_eff = eta_sum/kappa_sum;
			cov[i] = 1.0*(1.0/(1 + pow(eta_eff, -hillc)) > thresh);
		}
		else
		{
			double eta_sum = 0.0;
			double kappa_sum = 1.0;
			if (chosenclass > 0.5) 
			{
				eta_sum += ct*eta_t[currtarget][i]/kappa_t[currtarget][i];
				kappa_sum += ct*1.0/kappa_t[currtarget][i];
			}
			for(int j = 0 ;j < nb_temp; j++) 
			{
				eta_sum += cb[j]*eta_t[chosen[j]][i]/kappa_t[chosen[j]][i];
				kappa_sum += cb[j]*1.0/kappa_t[chosen[j]][i];
			}
			double eta_eff = eta_sum/kappa_sum;
			cov[i] = 1.0*(1.0/(1 + pow(eta_eff, -hillc)) > thresh);
		}
	}
	return cov;
}

vector<double> generate_sample_selected(double * chosen, int numt, const vector<vector<double> > & kappa_t,const vector<vector<double> > & eta_t)
{
	vector<double> cov(N,0.0);
	vector<double> cb(numt);
	
	if(mode == 1)
	{
		for(int l = 0 ; l < numt; l++)
		{
			//cb[l] = -exp(lnc_b)*log(gsl_rng_uniform_pos(randgen)); 
			cb[l] = exp(log(10)*lnc_t*(gsl_rng_uniform_pos(randgen) - 0.5 +  5*log(10)));
		}
	}

	for(int i = 0 ; i < N; i++)
	{
		if(mode == 0)
		{
			double eta_sum = 0;
			double kappa_sum = 0;
			for(int j = 0 ;j < numt; j++) 
			{
				eta_sum += eta_t[chosen[j]][i]/kappa_t[chosen[j]][i];
				kappa_sum += 1.0/kappa_t[chosen[j]][i];
			}
			double eta_eff = eta_sum/kappa_sum;
			cov[i] = 1.0*(1.0/(1 + pow(eta_eff, -hillc)) > thresh);
		}
		else
		{
			double eta_sum = 0.0;
			double kappa_sum = 1.0;
			for(int j = 0 ;j < numt; j++) 
			{
				eta_sum += cb[j]*eta_t[chosen[j]][i]/kappa_t[chosen[j]][i];
				kappa_sum += cb[j]*1.0/kappa_t[chosen[j]][i];
			}
			double eta_eff = eta_sum/kappa_sum;
			cov[i] = 1.0*(1.0/(1 + pow(eta_eff, -hillc)) > thresh);
		}
	}
	return cov;
}

vector<double> generate_sample_overshadow(double * chosen,int numt, double c1, double c2, const vector<vector<double> > & kappa_t,const vector<vector<double> > & eta_t)
{
	vector<double> cov(N,0.0);
	vector<double> cb(numt);
	
	if(mode == 1)
	{
		cb[0] = c1;
		cb[1] = c2;
	}

	for(int i = 0 ; i < N; i++)
	{
		if(mode == 0)
		{
			double eta_sum = 0;
			double kappa_sum = 0;
			for(int j = 0 ;j < numt; j++) 
			{
				eta_sum += eta_t[chosen[j]][i]/kappa_t[chosen[j]][i];
				kappa_sum += 1.0/kappa_t[chosen[j]][i];
			}
			double eta_eff = eta_sum/kappa_sum;
			cov[i] = 1.0*(1.0/(1 + pow(eta_eff, -hillc)) > thresh);
		}
		else
		{
			double eta_sum = 0.0;
			double kappa_sum = 1.0;
			for(int j = 0 ;j < numt; j++) 
			{
				eta_sum += cb[j]*eta_t[chosen[j]][i]/kappa_t[chosen[j]][i];
				kappa_sum += cb[j]*1.0/kappa_t[chosen[j]][i];
			}
			double eta_eff = eta_sum/kappa_sum;
			cov[i] = 1.0*(1.0/(1 + pow(eta_eff, -hillc)) > thresh);
		}
	}
	return cov;
}

int main(int argc, char* argv[])
{
	gsl_rng_set(randgen, time(NULL));
	int numiter = numtrain;
	int hiddenlayer_numnodes = 100;
	int act_func2 = 1;
	int numtrials = 5;

	if(argc > 1) numiter = atoi(argv[1]);
	if(argc > 2) numtrain = atoi(argv[2]);
	if(argc > 3) alpha = atof(argv[3]);
	if(argc > 4) mu = atof(argv[4]);
	if(argc > 5) lamb = atof(argv[5]);
	if(argc > 6) rho = atof(argv[6]);
	if(argc > 7) sparsity = atof(argv[7]);
	if(argc > 8) numtarget = atoi(argv[8]);
	if(argc > 9) numtrials = atoi(argv[9]);
	if(argc > 10) mode = atoi(argv[10]);
	if(argc > 11) lnc_t = atof(argv[11]);
	if(argc > 12) max_numt = atoi(argv[12]);
	if(argc > 13) thresh_classifier = atof(argv[13]);
	if(argc > 14) c_ratio = atof(argv[14]);

	for(int i = 1 ; i < argc; i++ ) cout << argv[i] << " " ; 
	cout << endl;

	numclasses = 2;

	thresh = 1.0/(1.0 + exp(-hillc*gsl_cdf_ugaussian_Qinv(sparsity)));

	double avg_fpos = 0;
	double avg_hit = 0;

	vector<double> learned_weights(N);
	double learned_bias = 0;
	for(int num = 0; num < numtrials; num++)
	{
		vector<vector<double> > kappa_t(numtarget,vector<double>(N,0.0));
		vector<vector<double> > eta_t(numtarget,vector<double>(N,0.0));
		vector<double> lnkappa(N,0.0);
		vector<double> lneta(N,0.0);

		for(int m = 0 ; m < numtarget; m++)
		{
			for(int i = 0; i < N; i++)
			{
				lnkappa[i] = gsl_ran_gaussian(randgen,1.0); 
				lneta[i] = gsl_ran_gaussian(randgen,1.0); 

				lneta[i] = rho*lnkappa[i] + sqrt(1 - rho*rho)*lneta[i];
			}
			for(int i = 0; i < N; i++)
			{
				kappa_t[m][i] = exp(-4.0*lnkappa[i]); 
				eta_t[m][i] =  exp(lneta[i]);
			}
		}

		//generate train and test set
		

		

		double targetbag[numtarget];
		for(int i =0 ; i < numtarget; i++) targetbag[i] = (double) i ; 

		//initializing classifier
		vector<softmax> classifier;
		for(int t= 0 ;t < numtarget; t++)
		{
			softmax temp_classifier;
			temp_classifier.initialize_predictor(numclasses, N);
			temp_classifier.lin_an = 0;
			vector<vector<double> > train(numtrain, vector<double>(N,0.0));
			vector<vector<double> > trainout(numtrain, vector<double>(numclasses,0.0));
			for(int n = 0; n < numtrain; n++)
			{
				int chosenclass = floor(numclasses*gsl_rng_uniform_pos(randgen));
				trainout[n][chosenclass] = 1;
				train[n] = generate_sample(chosenclass, t,  kappa_t, eta_t);
			}
			classifier.push_back(temp_classifier);
			double power = 1.5;
			
			for(int i = 0 ; i < log(numiter)/log(1.5); i++)
			{
				classifier[t].train_SGD_predictor(train, trainout, alpha, mu, lamb, ceil(pow(power, i+1)) - ceil(pow(power, i)) );
				//perf = 100.0*classifier.test_predictor(test,testout) ;
				//cout << ceil(pow(power, i+1)) - 1  <<  " " <<  perf << endl;	
			}			
		}
		vector<vector<int> > test_all(numtest, vector<int> (numtarget,0));
		vector<vector<int> > testout_all(numtest, vector<int> (numtarget,0));
		double hitrate_avg = 0;
		double fprate_avg = 0;
		for(int n = 0; n < numtest; n++)
		{
			vector<vector<vector<double> > >test(numtarget, vector<vector<double> >(1, vector<double>(N,0.0)));
			vector<vector<vector<double> > >testout(numtarget, vector<vector<double> >(1, vector<double>(numclasses,1)));
			
			for(int  t= 0 ; t < numtarget; t++) testout[t][0][1] = 0;
			int numt = 2;
			double * chosen = new double[numt];
			gsl_ran_choose(randgen, chosen, numt, targetbag, numtarget, sizeof(double));
			for(int i = 0 ; i < numt; i++) 
			{
				testout[(int) chosen[i]][0][1]= 1; 
				testout[(int) chosen[i]][0][0]= 0;
				test_all[n][(int) chosen[i]] = 1;
			}
			double c1 = exp(10*log(10));
			double c2 = exp(10*log(10) + c_ratio*log(10));
			vector<double> test_sample = generate_sample_overshadow(chosen, numt , c1 , c2, kappa_t, eta_t);
			double hitrate = 0;
			double fprate = 0;
			for(int t = 0 ;t < numtarget; t++)
			{
				vector<double> prob(numclasses);
				prob = classifier[t].evaluate_prob(test_sample);
				bool presence = 0;
				if(prob[1] > thresh_classifier) presence = 1;
				double err = classifier[t].test_predictor(test[t],testout[t]);
				testout_all[n][t] = presence;
				if(test_all[n][t] == 1 && testout_all[n][t] == 1) hitrate += 1.0;
				if(test_all[n][t] == 0 && testout_all[n][t] == 1) fprate += 1.0;
			}

			for(int i = 0 ; i < numt; i++)
			{
				vector<double> prob(numclasses);
				prob = classifier[(int)chosen[i]].evaluate_prob(test_sample);
				cout << prob[1] << " " ; 
			}
			cout << endl;

			hitrate_avg += hitrate/numt;
			fprate_avg += fprate;
		}
		hitrate_avg /= numtest;
		fprate_avg /= numtest;

		avg_fpos += fprate_avg;
		avg_hit += hitrate_avg;
	}
	avg_fpos /= numtrials;
	avg_hit /= numtrials;
		
	//cout << avg_hit << " " << avg_fpos << endl;

	gsl_rng_free(randgen);
	return 0;
}