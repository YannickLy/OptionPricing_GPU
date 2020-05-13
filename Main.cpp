#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "kernel.cuh"

using namespace std;

// méthode pour pricer une option d'achat de manière séquentielle avec le CPU et sans parallélisation.
double mc_call_CPU(float T, float K, float S0, float sigma, float mu, float r, float dt, vector<float> h_normals, unsigned N_STEPS, unsigned N_PATHS) {

	double payoff_CPU = 0.0;
	float s_curr = 0.0;
	for (size_t i = 0; i < N_PATHS; i++) {
		int n_idx = i * N_STEPS;
		s_curr = S0;
		int n = 0;
		do {
			s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * h_normals[n_idx];
			n_idx++;
			n++;
		} while (n < N_STEPS);
		double payoff = (s_curr > K ? s_curr - K : 0.0);
		payoff_CPU += exp(-r * T) * payoff;
	}
	return (payoff_CPU /= N_PATHS);

}

// méthode pour pricer une option de vente de manière séquentielle avec le CPU et sans parallélisation.
double mc_put_CPU(float T, float K, float S0, float sigma, float mu, float r, float dt, vector<float> h_normals, unsigned N_STEPS, unsigned N_PATHS) {

	double payoff_CPU = 0.0;
	float s_curr = 0.0;
	for (size_t i = 0; i < N_PATHS; i++) {
		int n_idx = i * N_STEPS;
		s_curr = S0;
		int n = 0;
		do {
			s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * h_normals[n_idx];
			n_idx++;
			n++;
		} while (n < N_STEPS);
		double payoff = (s_curr < K ? K - s_curr : 0.0);
		payoff_CPU += exp(-r * T) * payoff;
	}
	return (payoff_CPU /= N_PATHS);
}

int main() {

	int choix;
	char choix2;

	size_t N_PATHS = 100000;
	size_t N_STEPS = 365;
	size_t N_NORMALS = N_PATHS*N_STEPS;
	float T = 1.0f;
	float K = 100.0f;
	float S0 = 100.0f;
	float sigma = 0.2f;
	float mu = 0.1f;
	float r = 0.01f;
	float dt = float(T) / float(N_STEPS);
	float sqrdt = sqrt(dt);

	double price_GPU = 0.0; double price_CPU = 0.0;
	double t1 = 0.0; double t2 = 0.0; double t3 = 0.0; double t4 = 0.0;

	// Interface utilisateur
	cout << "Veuillez choisir le type d'option que vous souhaitez pricer :" << endl << endl;
	cout << "1. Option d'achat." << endl;
	cout << "2. Option de vente." << endl << endl;
	cout << "Votre choix 1-2 : "; cin >> choix; cout << endl;
	while (choix != 1 && choix != 2)
	{
		cout << "Le choix est incorrect, veuillez recommencer." << endl << endl;
		cout << "Veuillez choisir le type d'option que vous souhaitez pricer :" << endl << endl;
		cout << "1. Option d'achat." << endl;
		cout << "2. Option de vente." << endl << endl;
		cout << "Votre choix 1-2 : "; cin >> choix; cout << endl;
	}

	if (choix == 1)
	{
		cout << "Vous avez choisi l'option d'achat." << endl << endl;
	}
	else if (choix == 2)
	{
		cout << "Vous avez choisi l'option de vente." << endl << endl;
	}

	cout << "Nombre de trajectoires a simuler (mettre 100 000 par defaut) : "; cin >> N_PATHS;
	cout << "Nombre de pas par trajectoire (mettre 365 par defaut) : "; cin >> N_STEPS;
	cout << "Prix du sous-jacent (mettre 100 par defaut) : "; cin >> S0;
	cout << "Strike (mettre 110 par defaut) : "; cin >> K;
	cout << "Taux sans risque (mettre 0.01 par defaut) : "; cin >> r;
	cout << "Volatilite (mettre 0.2 par defaut) : "; cin >> sigma;
	cout << "Maturite en annees (mettre 1 par defaut) : "; cin >> T;
	cout << "Drift annuel : (mettre 0.1 par defaut) : "; cin >> mu; cout << endl;

	dt = float(T) / float(N_STEPS);
	sqrdt = sqrt(dt);

	vector<float> h_S(N_PATHS);
	float *d_S; // device pour stocker les prix du sous jacent simulés sur tous les chemins
	float *d_normals; // device pour stocker des chiffres aléatoires générés de manière gaussienne

	cudaMalloc((void**)&d_S, N_PATHS * sizeof(float)); // allocation dans la mémoire du GPU
	cudaMalloc((void**)&d_normals, N_NORMALS * sizeof(float)); // allocation dans la mémoire du GPU

	// Génération d'un vecteur de nombre aléatoire généré de manière gaussienne centrée et de volatilité "sqrdt" grâce à CUrand de CUDA
	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);
	curandGenerateNormal(curandGenerator, d_normals, N_NORMALS, 0.0f, sqrdt); // Sauvegarde du vecteur dans la mémoire GPU

	if (choix == 1)
	{
		t1 = double(clock()) / CLOCKS_PER_SEC;
		mc_call_GPU(d_S, T, K, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS); // appel du wrapper "mc_call_GPU" qui appelle une fonction __global__ CUDA
		cudaDeviceSynchronize(); // on attend que tous les threads aient finis leur calcul pour repasser en calcul CPU

		cudaMemcpy(&h_S[0], d_S, N_PATHS * sizeof(float), cudaMemcpyDeviceToHost); // on repasse de la mémoire GPU à la mémoire CPU les prix simulés

		for (size_t i = 0; i < N_PATHS; i++) {
			price_GPU += h_S[i];
		}
		price_GPU /= N_PATHS;
		t2 = double(clock()) / CLOCKS_PER_SEC;

		vector<float> h_normals(N_NORMALS);
		cudaMemcpy(&h_normals[0], d_normals, N_NORMALS * sizeof(float), cudaMemcpyDeviceToHost); // on repasse le vecteur de nombre aléatoire de la mémoire GPU à la mémoire CPU afin qu'on puisse l'utiliser pour le codé séquentiel

		t3 = double(clock()) / CLOCKS_PER_SEC;
		price_CPU = mc_call_CPU(T, K, S0, sigma, mu, r, dt, h_normals, N_STEPS, N_PATHS);
		t4 = double(clock()) / CLOCKS_PER_SEC;
	}
	else if (choix == 2)
	{
		t1 = double(clock()) / CLOCKS_PER_SEC;
		mc_put_GPU(d_S, T, K, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
		cudaDeviceSynchronize();

		cudaMemcpy(&h_S[0], d_S, N_PATHS * sizeof(float), cudaMemcpyDeviceToHost);

		for (size_t i = 0; i < N_PATHS; i++) {
			price_GPU += h_S[i];
		}
		price_GPU /= N_PATHS;
		t2 = double(clock()) / CLOCKS_PER_SEC;

		vector<float> h_normals(N_NORMALS);
		cudaMemcpy(&h_normals[0], d_normals, N_NORMALS * sizeof(float), cudaMemcpyDeviceToHost);

		t3 = double(clock()) / CLOCKS_PER_SEC;
		price_CPU = mc_put_CPU(T, K, S0, sigma, mu, r, dt, h_normals, N_STEPS, N_PATHS);
		t4 = double(clock()) / CLOCKS_PER_SEC;
	}
	
	cudaFree(d_S);
	cudaFree(d_normals);

	// Résultats
	cout << "********************* INFO *********************" << endl;
	cout << "Nombre de trajectoires a simuler : " << N_PATHS << endl;
	cout << "Nombre de pas par trajectoire : " << N_STEPS << endl;
	cout << "Prix du sous-jacent : " << S0 << endl;
	cout << "Strike	: " << K << endl;
	cout << "Taux sans risque : " << r << endl;
	cout << "Volatilite : " << sigma << endl;
	cout << "Maturite en annees : " << T << endl;
	cout << "Drift annuel : " << mu << endl;
	cout << "********************* PRICE *********************" << endl;
	cout << "Prix de l'option (GPU) : " << price_GPU << endl;
	cout << "Prix de l'option (CPU) : " << price_CPU << endl;
	cout << "********************* TEMPS D'EXECUTION *********************" << endl;
	cout << "GPU Monte Carlo Computation : " << (t2 - t1)*1e3 << " ms\n";
	cout << "CPU Monte Carlo Computation : " << (t4 - t3)*1e3 << " ms\n";

	system("pause");
	return 0;
}

