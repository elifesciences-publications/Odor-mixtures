#include "common.hpp"

int N = 250;
double rho = 0.0;
int numb = 10;
int hillc = 4;
int numtrials = 10;
int numsamples = 1000;
int numsamples_I = 1000;
double sparsity = 0.5;
double tau;


vector<double> prob_nb;

gsl_rng * randgen = gsl_rng_alloc(gsl_rng_taus2);


vector<bool> generate_sample(int size, vector<double> kappa_t, vector<double> eta_t)
{
	vector<bool>  output(2*N);

	vector<vector<double> > eta(size, vector<double>(N));
	vector<vector<double> > kappa(size, vector<double>(N));

	for(int l  = 0 ; l < size; l++)
	{
		vector<double> lneta(N,0);
		vector<double> lnkappa(N,0);
		for(int j = 0 ; j < N ; j++)
		{
			double lneta = gsl_ran_gaussian(randgen,1.0);	
			double lnkappa = gsl_ran_gaussian(randgen,1.0);	

			lneta = rho*lnkappa + sqrt(1 - rho*rho)*lneta;
			lnkappa = -4*lnkappa; 
			
			eta[l][j] = exp(lneta);
			kappa[l][j] = exp(lnkappa);
		}
	}

	for(int j = 0 ; j < N; j++)
	{
		double kappa_sum = 0.0;
		double etakappa_sum  = 0.0;
		double kappa_sum_t = 0.0;
		double etakappa_sum_t  = 0.0;
		for(int l = 0 ; l < size ; l++)
		{
			if(l == 0)
			{
				kappa_sum_t += 1.0/kappa_t[j];
				etakappa_sum_t += eta_t[j]/kappa_t[j];
			}
			else
			{
				kappa_sum_t += 1.0/kappa[l][j]; 
				etakappa_sum_t += eta[l][j]/kappa[l][j];	
			}

			kappa_sum += 1.0/kappa[l][j]; 
			etakappa_sum += eta[l][j]/kappa[l][j];		
		}
		double eta_eff = etakappa_sum/kappa_sum;
		double eta_eff_t = etakappa_sum_t/kappa_sum_t; 		
		double f_eff = 1.0/(1.0 + pow(eta_eff,-hillc));
		double f_eff_t = 1.0/(1.0 + pow(eta_eff_t,-hillc));	
		output[j] = f_eff > tau;
		output[N+j] = f_eff_t > tau;
	}
	return output;
}

double estimate_mutualinfo(vector<double> ps, vector<double>  ps_t, vector<double> kappa_t, vector<double> eta_t)
{
	double temp = 0.0 ;
	for(int  i = 0 ; i < numsamples_I ; i++)
	{
		vector<bool> z(N);

		double ll_t = 0.0;
		double ll = 0.0;

		int flag = 0;
		double p_t ;
		if(gsl_rng_uniform_pos(randgen) > 0.5)
		{
			flag = 1;
		}
		else flag = 0;
		vector<bool>  sample_output(2*N);
		sample_output = generate_sample(numb,kappa_t, eta_t);


		for(int j = 0 ; j < N; j++)
		{
			z[j] = flag == 1 ? sample_output[N+j] : sample_output[j];
			ll_t += log(z[j]*ps_t[j] + (1-z[j])*(1-ps_t[j]) + 1e-14);
			ll += log(z[j]*ps[j] + (1-z[j])*(1-ps[j]) + 1e-14);
		}
		p_t = 1.0/(1.0 + exp(ll - ll_t));
		temp += 1.0 - (-(p_t + 1e-14)*log2((p_t + 1e-14)) - (1 - p_t + 1e-14)*log2(1 - p_t + 1e-14));
	}
	return temp/numsamples_I;
}

int main(int argc, char* argv[])
{
	gsl_rng_set(randgen, time(NULL));

	if(argc > 1) numb = atoi(argv[1]);
	if(argc > 2) rho = atof(argv[2]);
	if(argc > 3) sparsity = atof(argv[3]);
	if(argc > 4) numtrials = atoi(argv[4]);
	if(argc > 5) numsamples = atoi(argv[5]);
	if(argc > 6) numsamples_I = atoi(argv[6]);

	tau = 1.0/(1.0 + exp(-hillc*gsl_cdf_ugaussian_Qinv(sparsity)));


	double ent_avg = 0;

	for(int i = 0 ;i < numtrials; i++)
	{
		vector<double> lneta_t(N);
		vector<double> lnkappa_t(N);

		vector<double> kappa_t(N);
		vector<double> eta_t(N);

		for(int j = 0 ; j < N ; j++)
		{
			lneta_t[j] = gsl_ran_gaussian(randgen,1.0);	
			lnkappa_t[j] = gsl_ran_gaussian(randgen,1.0);	

			lneta_t[j] = rho*lnkappa_t[j] + sqrt(1 - rho*rho)*lneta_t[j];
			lnkappa_t[j] = -4*lnkappa_t[j];

			eta_t[j] = exp(lneta_t[j]);
			kappa_t[j] = exp(lnkappa_t[j]);
		
		}

		vector<double> ps(N,0);
		vector<double> ps_t(N,0);
	

		for(int k = 0 ; k < numsamples; k++)
		{
			vector<bool>  sample_output(2*N);
			sample_output = generate_sample(numb,kappa_t, eta_t);
			for(int j = 0 ; j < N; j++)
			{
				ps[j] += (1.0*sample_output[j]/numsamples);
				ps_t[j] += (1.0*sample_output[N+j]/numsamples);
			}
		}	

		double I = estimate_mutualinfo(ps,ps_t, kappa_t, eta_t);
		ent_avg += I;
	}
	for(int i = 1 ; i < argc; i++ ) cout << argv[i] << " " ; 
	cout << endl;
	cout << ent_avg/numtrials << endl;
	return 0;
}