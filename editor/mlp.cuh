#pragma once

// Engine headers
#include "includa/cuda/math.cuh"

template <size_t N, size_t M>
struct Matrix {
	float *weights = nullptr;
	float *biases = nullptr;
};

// Initializing weights
float xavier(size_t n, size_t m)
{
	return sqrtf(6.0f / (n + m));
}

template <size_t N, size_t M>
Matrix <N, M> dev_matrix()
{
	float *weights = new float[N * M];
	float *biases = new float[N];

	for (size_t i = 0; i < N * M; i++)
		weights[i] = xavier(N, M) * (rand() / (float) RAND_MAX - 0.5f);

	for (size_t i = 0; i < N; i++)
		biases[i] = xavier(N, M) * (rand() / (float) RAND_MAX - 0.5f);

	Matrix <N, M> matrix;

	cudaMalloc(&matrix.weights, N * M * sizeof(float));
	cudaMalloc(&matrix.biases, N * sizeof(float));

	cudaMemcpy(matrix.weights, weights, N * M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(matrix.biases, biases, N * sizeof(float), cudaMemcpyHostToDevice);

	return matrix;
}

template <size_t N, size_t M>
void release(Matrix <N, M> matrix)
{
	cudaFree(matrix.weights);
	cudaFree(matrix.biases);
}

template <size_t N, size_t M, size_t B>
__global__
void forward(Matrix <N, M> matrix, Matrix <M, B> batch, Matrix <N, B> output)
{
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t j = threadIdx.y + blockIdx.y * blockDim.y;

	const size_t stride_x = blockDim.x * gridDim.x;
	const size_t stride_y = blockDim.y * gridDim.y;

	while (i < N && j < B) {
		float sum = 0.0f;
		for (size_t k = 0; k < M; k++)
			sum += matrix.weights[i * M + k] * batch.weights[k * B + j];
		output.weights[i * B + j] = sum + matrix.biases[i];

		i += stride_x;
		j += stride_y;
	}
}

// Radiance field neural network
struct RadianceField {
	// TODO: use positional encoding later
	constexpr static size_t IN = 3;
	constexpr static size_t OUT = 3;
	constexpr static size_t HIDDEN = 256;

	Matrix <IN, HIDDEN> D1;
	Matrix <HIDDEN, HIDDEN> D2;
	Matrix <HIDDEN, HIDDEN> D3;
	Matrix <HIDDEN, HIDDEN> D4;
	Matrix <HIDDEN, OUT> D5;

	struct {
		Matrix <IN, HIDDEN> D1;
		Matrix <HIDDEN, HIDDEN> D2;
		Matrix <HIDDEN, HIDDEN> D3;
		Matrix <HIDDEN, HIDDEN> D4;
		Matrix <HIDDEN, OUT> D5;
	} grad;

	struct {
		Matrix <HIDDEN, HIDDEN> D1;
		Matrix <HIDDEN, HIDDEN> D2;
		Matrix <HIDDEN, HIDDEN> D3;
		Matrix <HIDDEN, HIDDEN> D4;
	} cache;
};

// Initializing the neural network
RadianceField dev_radiance_field()
{
	RadianceField field;
	field.D1 = dev_matrix <RadianceField::IN, RadianceField::HIDDEN> ();
	field.D2 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();
	field.D3 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();
	field.D4 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();
	field.D5 = dev_matrix <RadianceField::HIDDEN, RadianceField::OUT> ();

	field.grad.D1 = dev_matrix <RadianceField::IN, RadianceField::HIDDEN> ();
	field.grad.D2 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();
	field.grad.D3 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();
	field.grad.D4 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();
	field.grad.D5 = dev_matrix <RadianceField::HIDDEN, RadianceField::OUT> ();

	field.cache.D1 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();
	field.cache.D2 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();
	field.cache.D3 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();
	field.cache.D4 = dev_matrix <RadianceField::HIDDEN, RadianceField::HIDDEN> ();

	return field;
}

void release(RadianceField field)
{
	release(field.D1);
	release(field.D2);
	release(field.D3);
	release(field.D4);
	release(field.D5);

	release(field.grad.D1);
	release(field.grad.D2);
	release(field.grad.D3);
	release(field.grad.D4);
	release(field.grad.D5);

	release(field.cache.D1);
	release(field.cache.D2);
	release(field.cache.D3);
	release(field.cache.D4);
}

// Forward pass of the neural network
template <size_t B = 1024>
void forward(RadianceField field, Matrix <RadianceField::IN, B> input, Matrix <RadianceField::OUT, B> output)
{
	dim3 threads(32, 32);
	dim3 blocks((RadianceField::OUT + threads.x - 1) / threads.x, (B + threads.y - 1) / threads.y);

	forward <<<blocks, threads>>> (field.D1, input, field.cache.D1);
	forward <<<blocks, threads>>> (field.D2, field.cache.D1, field.cache.D2);
	forward <<<blocks, threads>>> (field.D3, field.cache.D2, field.cache.D3);
	forward <<<blocks, threads>>> (field.D4, field.cache.D3, field.cache.D4);
	forward <<<blocks, threads>>> (field.D5, field.cache.D4, output);
}
