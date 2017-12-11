#ifndef _FFNN_HPP_
#define _FFNN_HPP_

#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include <stdlib.h>
using namespace std;

namespace ffnn {
        default_random_engine generator(11);

	template <typename T>
		class ffnn_node {
			public:
				T* weights;
				int nweights;

				ffnn_node(int ninputs) : nweights(ninputs + 1), weights(new T[ninputs + 1]) {
				}

				~ffnn_node() {
					delete [] weights;
				}

				void randomWeightsNormal(double mean, double stddev) {
					normal_distribution<double> distribution(mean, stddev);

					for (int i = 0; i < nweights; ++i) {
						weights[i] = distribution(generator);
					}
				}

				void randomWeightsUniform(double lower, double upper) {
					uniform_real_distribution<double> distribution(lower, upper);

					for (int i = 0; i < nweights; ++i) {
						weights[i] = distribution(generator);
					}
				}

				int getNWeights() {
					return nweights;
				}

				void getWeights(int &index, T **w) {
					for (int i = 0; i < nweights; ++i) {
						w[index++] = &weights[i];
					}
				}

				inline T evaluate(const T *x) {
					T sum = weights[0];

					for (int i = 1; i < nweights; ++i) {
						sum += weights[i] * x[i - 1];
					}

					return sum;
				}

				inline void evaluate_adjoint(const T* x, const T sum_a, T *x_a, int &index, T *w_a) {
					for (int i = nweights - 1; i >= 1; --i) {
						w_a[--index] += x[i - 1] * sum_a;
						x_a[i - 1] += weights[i] * sum_a;
					}

                                        w_a[--index] += sum_a;
				}

				void print() {
					for (int i = 0; i < nweights; ++i) {
						cout << weights[i] << " ";
					}
				}
		};

	template <typename T>
		class ffnn_layer {
			public:
				ffnn_node<T>** nodes;
				int nnodes;
				int ninputs;

				T *values;
				T *values_a;

				function<T(T)> activation_function;
				function<T(T)> activation_function_derivative;

				ffnn_layer(int nthreads, int ninputs, int nnodes, function<T(T)> activation_fun, function<T(T)> activation_fun_der) : ninputs(ninputs), nnodes(nnodes), nodes(new ffnn_node<T>*[nnodes]), values(new T[nthreads * nnodes]), values_a(new T[nthreads * ninputs]) {
					activation_function = activation_fun;
					activation_function_derivative = activation_fun_der;

					for (int i = 0; i < nnodes; ++i) {
						nodes[i] = new ffnn_node<T>(ninputs);
					}
				}

				~ffnn_layer() {
					for (int i = 0; i < nnodes; ++i) {
						delete nodes[i];
					}
					delete [] nodes;

					delete [] values;
					delete [] values_a;
				}

				void randomWeightsNormal(double mean, double stddev) {
					for (int i = 0; i < nnodes; ++i) {
						nodes[i]->randomWeightsNormal(mean, stddev);
					}
				}

				void randomWeightsUniform(double lower, double upper) {
					for (int i = 0; i < nnodes; ++i) {
						nodes[i]->randomWeightsUniform(lower, upper);
					}
				}

				int getNWeights() {
					int sum = 0;
					for (int i = 0; i < nnodes; ++i) {
						sum += nodes[i]->getNWeights();
					}
					return sum;
				}

				void getWeights(int &index, T **w) {
					for (int i = 0; i < nnodes; ++i) {
						nodes[i]->getWeights(index, w);
					}
				}

				inline void evaluate(int threadn, const T *x) {
					for (int i = 0; i < nnodes; ++i) {
                                                T sum = nodes[i]->evaluate(x);
                                                values[threadn * nnodes + i] = activation_function(sum);
					}
				}

				inline void evaluate_adjoint(int threadn, const T *x, const T *y_a, int &index, T *w_a) {
					for (int i = 0; i < ninputs; ++i) {
						values_a[threadn * ninputs + i] = 0.0;
					}

					for (int i = nnodes - 1; i >= 0; --i) {
                                                T sum = nodes[i]->evaluate(x);

                                                T sum_a = activation_function_derivative(sum) * y_a[i];
                                                nodes[i]->evaluate_adjoint(x, sum_a, &values_a[threadn * ninputs], index, w_a);
					}
				}

				void print() {
					for (int i = 0; i < nnodes; ++i) {
						cout << i << ": ";
						nodes[i]->print();
						cout << endl;
					}
				}
		};

	template <typename T>
		class ffnn_network {
			private:
				ffnn_network() {
				}

			public:
				ffnn_layer<T>** layers;
				int nlayers;

				ffnn_network(int nlayers) : nlayers(nlayers), layers(new ffnn_layer<T>*[nlayers]) {
				}

				~ffnn_network() {
					for (int i = 0; i < nlayers; ++i) {
						delete layers[i];
					}
					delete [] layers;
				}

				void setLayer(int index, ffnn_layer<T>* layer) {
					layers[index] = layer;
				}

				void randomWeightsNormal(double mean, double stddev) {
					for (int i = 0; i < nlayers; ++i) {
						layers[i]->randomWeightsNormal(mean, stddev);
					}
				}

				void randomWeightsUniform(double lower, double upper) {
					for (int i = 0; i < nlayers; ++i) {
						layers[i]->randomWeightsUniform(lower, upper);
					}
				}

				int getNWeights() {
					int sum = 0;
					for (int i = 0; i < nlayers; ++i) {
						sum += layers[i]->getNWeights();
					}
					return sum;
				}

				void getWeights(T **w) {
					int index = 0;
					for (int i = 0; i < nlayers; ++i) {
						layers[i]->getWeights(index, w);
					}
				}

                                inline void evaluate(int threadn, const double *x, T *y) {
                                        T *x_t1s = new T[layers[0]->ninputs];
                                        for (int i = 0; i < layers[0]->ninputs; ++i) {
                                            x_t1s[i] = x[i];
                                        }

                                        layers[0]->evaluate(threadn, x_t1s);

					for (int i = 1; i < nlayers; ++i) {
						layers[i]->evaluate(threadn, &layers[i - 1]->values[threadn * layers[i - 1]->nnodes]);
					}

					for (int i = 0; i < layers[nlayers - 1]->nnodes; ++i) {
						y[i] = layers[nlayers - 1]->values[threadn * layers[nlayers - 1]->nnodes + i];
					}

                                        delete [] x_t1s;
				}

                                inline void evaluate_adjoint(int threadn, const double* x, const T *y_a, T *w_a) {
                                        T *x_t1s = new T[layers[0]->ninputs];
                                        for (int i = 0; i < layers[0]->ninputs; ++i) {
                                            x_t1s[i] = x[i];
                                        }

					int index = getNWeights();

					if (nlayers == 1) {
                                                layers[nlayers - 1]->evaluate_adjoint(threadn, x_t1s, y_a, index, w_a);
                                                delete [] x_t1s;
						return;
					} 

					layers[nlayers - 1]->evaluate_adjoint(threadn, &layers[nlayers - 2]->values[threadn * layers[nlayers - 2]->nnodes], y_a, index, w_a);

					for (int i = nlayers - 2; i >= 1; --i) {
						layers[i]->evaluate_adjoint(threadn, &layers[i - 1]->values[threadn * layers[i - 1]->nnodes], &layers[i + 1]->values_a[threadn * layers[i + 1]->ninputs], index, w_a);
					}

                                        layers[0]->evaluate_adjoint(threadn, x_t1s, &layers[1]->values_a[threadn * layers[1]->ninputs], index, w_a);

                                        delete [] x_t1s;
                                }

				void print() {
					for (int i = 0; i < nlayers; ++i) {
						cout << "Layer " << i << endl;
						layers[i]->print();
						cout << endl;
					}
				}
		};
}

#endif
