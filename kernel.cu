#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "kernel.cuh"

__global__ void mc_kernel_call(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS)
{
	const unsigned tid = threadIdx.x; // id du thread dans le bloc
	const unsigned bid = blockIdx.x; // id du bloc
	const unsigned bsz = blockDim.x; // taille du bloc

	int s_idx = tid + bid * bsz;
	int n_idx = tid + bid * bsz;
	float s_curr = S0;

	if (s_idx < N_PATHS) {
		int n = 0;
		do {
			s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*d_normals[n_idx];
			n_idx++;
			n++;
		} while (n < N_STEPS);
		double payoff = (s_curr>K ? s_curr - K : 0.0);
		__syncthreads(); // on attend que tous les threads aient fini avant de passer à la prochaine simulation
		d_s[s_idx] = exp(-r*T) * payoff;
	}
}

__global__ void mc_kernel_put(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS)
{
	const unsigned tid = threadIdx.x;
	const unsigned bid = blockIdx.x;
	const unsigned bsz = blockDim.x;

	int s_idx = tid + bid * bsz;
	int n_idx = tid + bid * bsz;
	float s_curr = S0;

	if (s_idx < N_PATHS) {
		int n = 0;
		do {
			s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*d_normals[n_idx];
			n_idx++;
			n++;
		} while (n < N_STEPS);
		double payoff = (s_curr<K ? K - s_curr : 0.0);
		__syncthreads();
		d_s[s_idx] = exp(-r*T) * payoff;
	}
}

// wrapper pour une option d'achat
void mc_call_GPU(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS) 
{
	const unsigned BLOCK_SIZE = 1024; // utilisation de 1024 threads par bloc
	const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE)); // nombre de blocs nécessaires pour N_PATHS
	mc_kernel_call <<<GRID_SIZE, BLOCK_SIZE >>>(d_s, T, K, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS); // appel de la fonction parallélisée pour la simulation du prix du sous jacent
}

// wrapper pour une option de vente
void mc_put_GPU(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS)
{
	const unsigned BLOCK_SIZE = 1024;
	const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
	mc_kernel_put <<<GRID_SIZE, BLOCK_SIZE >> >(d_s, T, K, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}