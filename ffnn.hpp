#ifndef _FFNN_HPP_
#define _FFNN_HPP_

#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include <stdlib.h>
using namespace std;

namespace ffnn {
        default_random_engine generator(0);

	template <typename T>
	class ffnn_node {
		public:
			T* weights;
			unsigned int nweights;

			ffnn_node(unsigned int ninputs) : nweights(ninputs + 1), weights(new T[ninputs + 1]) {
			}

			~ffnn_node() {
				delete [] weights;
			}

			void randomWeightsNormal(double mean, double stddev) {
				normal_distribution<double> distribution(mean, stddev);

				for (unsigned int i = 0; i < nweights; ++i) {
					weights[i] = distribution(generator);
				}
			}

			void randomWeightsUniform(double lower, double upper) {
				uniform_real_distribution<double> distribution(lower, upper);

				for (unsigned int i = 0; i < nweights; ++i) {
					weights[i] = distribution(generator);
				}
			}

			unsigned int getNWeights() {
				return nweights;
			}

			void getWeights(unsigned int &index, T **w) {
				for (unsigned int i = 0; i < nweights; ++i) {
					w[index++] = &weights[i];
				}
			}

			inline T evaluate(const T *x) {
				T sum = weights[0];

				for (unsigned int i = 1; i < nweights; ++i) {
					sum += weights[i] * x[i - 1];
				}

				return sum;
			}

			void print() {
				for (unsigned int i = 0; i < nweights; ++i) {
					cout << weights[i] << " ";
				}
			}
	};

	template <typename T>
	class ffnn_layer {
		public:
			ffnn_node<T>** nodes;
			unsigned int nnodes;

			function<T(T)> activation_function;

			ffnn_layer(unsigned int ninputs, unsigned int nnodes, function<T(T)> activation_fun) : nnodes(nnodes), nodes(new ffnn_node<T>*[nnodes]) {
				activation_function = activation_fun;

				for (unsigned int i = 0; i < nnodes; ++i) {
					nodes[i] = new ffnn_node<T>(ninputs);
				}
			}

			~ffnn_layer() {
				for (unsigned int i = 0; i < nnodes; ++i) {
					delete nodes[i];
				}
				delete [] nodes;
			}

			void randomWeightsNormal(double mean, double stddev) {
				for (unsigned int i = 0; i < nnodes; ++i) {
					nodes[i]->randomWeightsNormal(mean, stddev);
				}
			}

			void randomWeightsUniform(double lower, double upper) {
				for (unsigned int i = 0; i < nnodes; ++i) {
					nodes[i]->randomWeightsUniform(lower, upper);
				}
			}

			unsigned int getNWeights() {
				unsigned int sum = 0;
				for (unsigned int i = 0; i < nnodes; ++i) {
					sum += nodes[i]->getNWeights();
				}
				return sum;
			}

			void getWeights(unsigned int &index, T **w) {
				for (unsigned int i = 0; i < nnodes; ++i) {
					nodes[i]->getWeights(index, w);
				}
			}

			inline void evaluate(const T *x, T *y) {
				for (unsigned int i = 0; i < nnodes; ++i) {
					y[i] = activation_function(nodes[i]->evaluate(x));
				}
			}

			void print() {
				for (unsigned int i = 0; i < nnodes; ++i) {
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
			unsigned int nlayers;

			ffnn_network(unsigned int nlayers) : nlayers(nlayers), layers(new ffnn_layer<T>*[nlayers]) {
			}

			~ffnn_network() {
				for (unsigned int i = 0; i < nlayers; ++i) {
					delete layers[i];
				}
				delete [] layers;
			}

			void setLayer(unsigned int index, ffnn_layer<T>* layer) {
				layers[index] = layer;
			}

			void randomWeightsNormal(double mean, double stddev) {
				for (unsigned int i = 0; i < nlayers; ++i) {
					layers[i]->randomWeightsNormal(mean, stddev);
				}
			}

			void randomWeightsUniform(double lower, double upper) {
				for (unsigned int i = 0; i < nlayers; ++i) {
					layers[i]->randomWeightsUniform(lower, upper);
				}
			}

			unsigned int getNWeights() {
				unsigned int sum = 0;
				for (unsigned int i = 0; i < nlayers; ++i) {
					sum += layers[i]->getNWeights();
				}
				return sum;
			}

			void getWeights(T **w) {
				unsigned int index = 0;
				for (unsigned int i = 0; i < nlayers; ++i) {
					layers[i]->getWeights(index, w);
				}
			}

                        inline void evaluate(const double *x, T *y) {
				T *tmp1, *tmp2;

                                unsigned int ninputs = layers[0]->nodes[0]->nweights - 1;
                                unsigned int noutputs = layers[nlayers - 1]->nnodes;

				tmp1 = new T[ninputs];
				for (unsigned int j = 0; j < ninputs; ++j) {
					tmp1[j] = x[j];
				}

				unsigned int i;
				for (i = 0; i < nlayers; ++i) {
					if (i % 2 == 0) {
						tmp2 = new T[layers[i]->nnodes];
						layers[i]->evaluate(tmp1, tmp2);
						delete [] tmp1;
					} else {
						tmp1 = new T[layers[i]->nnodes];
						layers[i]->evaluate(tmp2, tmp1);
						delete [] tmp2;
					}
				}

				if (i % 2 == 0) {
                                        for (unsigned int j = 0; j < noutputs; ++j) {
						y[j] = tmp1[j];
					}
					delete [] tmp1;
				} else {
                                        for (unsigned int j = 0; j < noutputs; ++j) {
						y[j] = tmp2[j];
					}
					delete [] tmp2;
				}
			}

			void print() {
				for (unsigned int i = 0; i < nlayers; ++i) {
					cout << "Layer " << i << endl;
					layers[i]->print();
					cout << endl;
				}
			}
	};
}

#endif
