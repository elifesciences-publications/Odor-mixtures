#include "softmax.hpp"


int N = 250;

int numtarget = 1;
int numclasses = numtarget + 1;
const int nummix = 100000;
int numbgs = 5;
int numtrain = 1000;
int numtest = 5000;

double alpha = 0.005;
double mu = 0.5;
double lamb = 1e-2;

double rho = 0.0;
double sparsity = 0.5;
int hillc = 4;
double thresh = 0.5;

double lnc_t = 0.0;
double lnc_b = 0.0;

double bgbag[nummix];

int mode = 0; // mode = 0 is infinite conc., in mode 1 you can vary it. 

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

vector<double> generate_sample(int chosenclass, const vector<vector<double> > & kappa_t,const vector<vector<double> > & eta_t,const vector<vector<double> > & kappa_b,const vector<vector<double> > & eta_b)
{
	vector<double> cov(N,0.0);
	int nb_temp = chosenclass > 0.5 ? 0 : 1 ;
	nb_temp += floor(exp(log(1.0*numbgs)*gsl_rng_uniform_pos(randgen)));

	double chosen[nb_temp];
	gsl_ran_choose(randgen, chosen, nb_temp, bgbag, nummix, sizeof(double));
	gsl_ran_shuffle(randgen, chosen, nb_temp, sizeof(double));

	
	vector<double> cb(nb_temp);
	//double ct = -exp(lnc_t)*log(gsl_rng_uniform_pos(randgen)); 
	double ct = exp(log(10)*3*(gsl_rng_uniform_pos(randgen) - 0.5 + lnc_t));
	
	if(mode == 1)
	{
		for(int l = 0 ; l < nb_temp; l++)
		{
			//cb[l] = -exp(lnc_b)*log(gsl_rng_uniform_pos(randgen)); 
			cb[l] = exp(log(10)*3*(gsl_rng_uniform_pos(randgen) - 0.5 + lnc_b));
		}
	}

	for(int i= 0 ; i < N; i++)
	{
		if(mode == 0)
		{
			double eta_sum = 0;
			double kappa_sum = 0;
			if (chosenclass > 0.5) 
			{
				eta_sum += eta_t[chosenclass-1][i]/kappa_t[chosenclass-1][i];
				kappa_sum += 1.0/kappa_t[chosenclass-1][i];
			}
			for(int j = 0 ;j < nb_temp; j++) 
			{
				eta_sum += eta_b[chosen[j]][i]/kappa_b[chosen[j]][i];
				kappa_sum += 1.0/kappa_b[chosen[j]][i];
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
				eta_sum += ct*eta_t[chosenclass-1][i]/kappa_t[chosenclass-1][i];
				kappa_sum += ct*1.0/kappa_t[chosenclass-1][i];
			}
			for(int j = 0 ;j < nb_temp; j++) 
			{
				eta_sum += cb[j]*eta_b[chosen[j]][i]/kappa_b[chosen[j]][i];
				kappa_sum += cb[j]*1.0/kappa_b[chosen[j]][i];
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
	if(argc > 8) numbgs = atoi(argv[8]);
	if(argc > 9) numtarget = atoi(argv[9]);
	if(argc > 10) numtrials = atoi(argv[10]);
	if(argc > 11) mode = atoi(argv[11]);
	if(argc > 12) lnc_t = atof(argv[12]);
	if(argc > 13) lnc_b = atof(argv[13]);
/*	if(argc > 8) c_mix = atof(argv[8])*c_t;
	if(argc > 9) numbgs = atoi(argv[9]);
	if(argc > 10) numtarget = atoi(argv[10]);
	if(argc > 11) numtrials = atoi(argv[11]);
	if(argc > 12) thresh_an = atof(argv[12]);
	if(argc > 13) mode  = atoi(argv[13]); //0 for lin, 1 for an
	if(argc > 14) hiddenlayer_numnodes = atoi(argv[14]);
	if(argc > 15) act_func2 = atoi(argv[15]);*/
	
	numclasses = numtarget + 1;
	
	thresh = 1.0/(1.0 + exp(-hillc*gsl_cdf_ugaussian_Qinv(sparsity)));

	double avg = 0;
	double std = 0 ;
	double avg_fpos = 0;
	double avg_fneg = 0;

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
			//if (m == 0) std::sort(lnkappa.begin(), lnkappa.end(), [](double i, double j) {return (i>j);});
			for(int i = 0; i < N; i++)
			{
				kappa_t[m][i] = exp(-4.0*lnkappa[i]); 
				eta_t[m][i] =  exp(lneta[i]);
			}
		}
		vector<vector<double> > kappa_b(nummix,vector<double>(N,0.0));
		vector<vector<double> > eta_b(nummix,vector<double>(N,0.0));
		//generating the distribution of binding constants for the target and background
		for(int m = 0 ; m < nummix; m++)
		{
			for(int i = 0; i < N; i++)
			{
				lnkappa[i] = gsl_ran_gaussian(randgen,1.0); 
				lneta[i] = gsl_ran_gaussian(randgen,1.0); 

				lneta[i] = rho*lnkappa[i] + sqrt(1 - rho*rho)*lneta[i];
			}
			//if (m == 0) std::sort(lnkappa.begin(), lnkappa.end(), [](double i, double j) {return (i>j);});
			for(int i = 0; i < N; i++)
			{
				kappa_b[m][i] = exp(-4.0*lnkappa[i]); 
				eta_b[m][i] =  exp(lneta[i]);
			}
		}
		//generate train and test set
		vector<vector<double> > train(numtrain, vector<double>(N,0.0));
		vector<vector<double> > trainout(numtrain, vector<double>(numclasses,0.0));

		vector<vector<double> > test(numtest, vector<double>(N,0.0));
		vector<vector<double> > testout(numtest, vector<double>(numclasses,0.0));

		for(int i = 0 ; i < nummix; i++) bgbag[i] = (double) i;

		for(int n = 0; n < numtrain; n++)
		{
			int chosenclass = floor(numclasses*gsl_rng_uniform_pos(randgen));
			trainout[n][chosenclass] = 1;
			train[n] = generate_sample(chosenclass, kappa_t, eta_t, kappa_b, eta_b);

		}

		int numpos = 0;
		int numneg = 0;
		for(int n = 0; n < numtest; n++)
		{
			int chosenclass = floor(numclasses*gsl_rng_uniform_pos(randgen));
			testout[n][chosenclass] = 1;
			test[n] =  generate_sample(chosenclass, kappa_t, eta_t, kappa_b, eta_b);
		}

		//initializing classifier
		softmax classifier;
		classifier.initialize_predictor(numclasses, N);
		classifier.lin_an = 0.0;
		double power = 1.5;
		double perf = 0;
		for(int i = 0 ; i < log(numiter)/log(1.5); i++)
		{
			classifier.train_SGD_predictor(train, trainout, alpha, mu, lamb, ceil(pow(power, i+1)) - ceil(pow(power, i)) );
			//perf = 100.0*classifier.test_predictor(test,testout) ;
			//cout << ceil(pow(power, i+1)) - 1  <<  " " <<  perf << endl;	
		}
		perf = 100.0*classifier.test_predictor(test,testout) ;
		ofstream fileout;
		fileout.open("failed.dat", ios::out | ios::trunc);
		int false_neg = 0;
		int false_pos = 0;
		for(int j = 0 ; j < classifier.failed_indices.size(); j++) 
		{
			if(testout[classifier.failed_indices[j]][1] > 0.5) false_neg += 1;
			else false_pos += 1;
		}
		fileout.close();

		for(int i = 0; i < N ; i++)
		{
			learned_weights[i] += classifier.c_theta[1][i] - classifier.c_theta[0][i];
		}
		learned_bias += classifier.c_bias[1] - classifier.c_bias[0];

		avg += perf;
		std += perf*perf;
		avg_fpos += false_pos*100.0/numneg;
		avg_fneg += false_neg*100.0/numpos;
		cout << false_pos << " " << false_neg << " " << numneg << " " << numpos <<  endl;

	}
	avg = avg/numtrials;
	avg_fpos = avg_fpos/numtrials;
	avg_fneg = avg_fneg/numtrials;
	std = sqrt(std/numtrials - avg*avg);
	for(int i = 1 ; i < argc; i++ ) cout << argv[i] << " " ; 
	cout << endl;	
	cout << avg << " " << std << " " << avg_fpos << " " << avg_fneg << endl;

	gsl_rng_free(randgen);
	return 0;
}