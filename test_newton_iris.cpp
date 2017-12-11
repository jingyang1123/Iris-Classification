#include <iomanip>
#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

#include <omp.h>
#include "dco_scrambled.hpp"
#include "ffnn_direct.hpp"

#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/IterativeLinearSolvers"

using namespace Eigen;
using Eigen::MatrixXd;
#define dtype double
#define ttype dco::gt1s<dtype>::type
const double epsilon = 0.1;

const int nthreads = 1;
ofstream csv;

class MatrixReplacement;
template<typename Rhs> class MatrixReplacement_ProductReturnType;
namespace Eigen {
namespace internal {
  template<>
  struct traits<MatrixReplacement> :  Eigen::internal::traits<Eigen::SparseMatrix<double> >
  {};
  template <typename Rhs>
  struct traits<MatrixReplacement_ProductReturnType<Rhs> > {
    // The equivalent plain objet type of the product. This type is used if the product needs to be evaluated into a temporary.
    typedef Eigen::Matrix<typename Rhs::Scalar, Eigen::Dynamic, Rhs::ColsAtCompileTime> ReturnType;
  };
}
}
// Inheriting EigenBase should not be needed in the future.
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement> {
public:
  // Expose some compile-time information to Eigen:
  typedef double Scalar;
  typedef double RealScalar;


  const dtype *x_input, *y_input;
  int number;
  dtype lambda;
  ffnn::ffnn_network<ttype> *test_network;
  int num_weights;

  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    RowsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    MaxRowsAtCompileTime = Eigen::Dynamic
  };
  Index rows() const { return num_weights; }
  Index cols() const { return num_weights; }
  void resize(Index a_rows, Index a_cols)
  {
    // This method should not be needed in the future.
    assert(a_rows==0 && a_cols==0 || a_rows==rows() && a_cols==cols());
  }
  // In the future, the return type should be Eigen::Product<MatrixReplacement,Rhs>
  template<typename Rhs>
  MatrixReplacement_ProductReturnType<Rhs> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return MatrixReplacement_ProductReturnType<Rhs>(x_input, y_input, number, lambda, test_network, num_weights, *this, x.derived());
  }
};
// The proxy class representing the product of a MatrixReplacement with a MatrixBase<>
template<typename Rhs>
class MatrixReplacement_ProductReturnType : public Eigen::ReturnByValue<MatrixReplacement_ProductReturnType<Rhs> > {
public:
  typedef MatrixReplacement::Index Index;


  const dtype *x_input, *y_input;
  int number;
  dtype lambda;
  ffnn::ffnn_network<ttype> *test_network;
  int num_weights;

  // The ctor store references to the matrix and right-hand-side object (usually a vector).
  MatrixReplacement_ProductReturnType( const dtype *x_array, const dtype *y_array, int num, dtype lam, ffnn::ffnn_network<ttype> *test_nw, int num_w, const MatrixReplacement& matrix, const Rhs& rhs)
    : x_input(x_array), y_input(y_array), number(num), lambda(lam), test_network(test_nw), num_weights(num_w), m_matrix(matrix), m_rhs(rhs)
  {}

  Index rows() const { return m_matrix.rows(); }
  Index cols() const { return m_rhs.cols(); }
  // This function is automatically called by Eigen. It must evaluate the product of matrix * rhs into y.

  template<typename Dest>
  void evalTo( Dest& f) const
  {
    typename dco::gt1s<dtype>::type *grad_t1s = new typename dco::gt1s<dtype>::type[num_weights];
    ttype eps = 1e-15;
    ttype** w_test = new ttype*[num_weights];
    test_network->getWeights(w_test);

    for ( int i = 0; i < num_weights; i++) {
        //cout << "w_test "<< i << ": "<< dco::value(*w_test[i]) << endl;
        dco::derivative(*w_test[i]) = m_rhs(i);

    }

    for(unsigned int i = 0; i < number; i++){
        ttype eval[3], eval_a[3], eval_m[3], p[3], p_a[3];
        test_network->evaluate(0, &x_input[i * 4], eval);
        ttype prob[3], prob_a[3];
        ttype m = eval[0], sum = 0.0, sum_a = 0.0;
        for (unsigned int j = 1; j < 3; j++){
            if (m < eval[j]){
                m = eval[j];
            }
        }
        for (unsigned int j = 0; j < 3; j++){
            eval_m[j] = eval[j] - m;
        }
        for (unsigned int j = 0; j < 3; j++){
            p[j] = exp(eval_m[j]);
            sum = sum + p[j];
        }
        for (unsigned int j = 0; j < 3; j++){
            ttype temp = prob[j] = p[j]/sum;
            prob[j] = max(prob[j], eps);

            prob_a[j] = -y_input[i * 3 + j] * (1 / prob[j]) * 1.0;
            if (temp < eps) {
                prob_a[j] = 0.0;
            }
            p_a[j] = (1 / sum) * prob_a[j];
            sum_a += -p[j] / (sum * sum) * prob_a[j];
        }
        for (int j = 2; j >= 0; j--){
            p_a[j] += 1.0 * sum_a;
        }
        for (unsigned int j = 0; j < 3; j++){
            eval_a[j] = p[j] * p_a[j];
        }
/*
        for (unsigned int j = 0; j < 10; j++){
            eval_a[j] = 2 * (eval[j] - y_input[i * 10 + j]);
        }
*/
        test_network->evaluate_adjoint(0, &x_input[i * 4], eval_a, grad_t1s);
    }

    f.setZero(num_weights);

    for ( int i = 0; i < num_weights; i++) {
        //cout << "grad_t1s " << i << ":" << grad_t1s[i] << " ";
        f(i) = dco::derivative(grad_t1s[i]) + lambda * m_rhs(i);

    }//cout << endl;

    delete [] grad_t1s;
  }
protected:
  const MatrixReplacement& m_matrix;
  typename Rhs::Nested m_rhs;
};
/*****/
// This class simply warp a diagonal matrix as a Jacobi preconditioner.
// In the future such simple and generic wrapper should be shipped within Eigen itsel.

template <typename _Scalar>
class MyJacobiPreconditioner
{
    typedef _Scalar Scalar;
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;
    typedef typename Vector::Index Index;
  public:
    // this typedef is only to export the scalar type and compile-time dimensions to solve_retval
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
    MyJacobiPreconditioner() : m_isInitialized(false) {}
    void setInvDiag(const Eigen::VectorXd &invdiag) {
      m_invdiag=invdiag;
      m_isInitialized=true;
    }
    Index rows() const { return m_invdiag.size(); }
    Index cols() const { return m_invdiag.size(); }

    template<typename MatType>
    MyJacobiPreconditioner& analyzePattern(const MatType& ) { return *this; }

    template<typename MatType>
    MyJacobiPreconditioner& factorize(const MatType& mat) { return *this; }

    template<typename MatType>
    MyJacobiPreconditioner& compute(const MatType& mat) { return *this; }
    template<typename Rhs, typename Dest>
    void _solve(const Rhs& b, Dest& x) const
    {
      x = m_invdiag.array() * b.array() ;
    }
    template<typename Rhs> inline const Eigen::internal::solve_retval<MyJacobiPreconditioner, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "MyJacobiPreconditioner is not initialized.");
      eigen_assert(m_invdiag.size()==b.rows()
                && "MyJacobiPreconditioner::solve(): invalid number of rows of the right hand side matrix b");
      return Eigen::internal::solve_retval<MyJacobiPreconditioner, Rhs>(*this, b.derived());
    }
  protected:
    Vector m_invdiag;
    bool m_isInitialized;
};
namespace Eigen {
namespace internal {
template<typename _MatrixType, typename Rhs>
struct solve_retval<MyJacobiPreconditioner<_MatrixType>, Rhs>
  : solve_retval_base<MyJacobiPreconditioner<_MatrixType>, Rhs>
{
  typedef MyJacobiPreconditioner<_MatrixType> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs);
  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec()._solve(rhs(),dst);
  }
};
}
}
/*****/


void ffnn_regression_newton(const unsigned int n, const dtype* x, const dtype* y, dtype alpha, ffnn::ffnn_network<ttype> *network, dtype lambda, const unsigned int trial) {
        double startt, endt1;
        startt = omp_get_wtime();

        unsigned int nweights = network->getNWeights();
        ttype** w = new ttype*[nweights];
        network->getWeights(w);
        dtype* delta = new dtype[nweights];
        ttype* gradient = new ttype[nweights];
        ttype eps = 1e-15;

        dtype lam = lambda;
        dtype res = 0.0;
        dtype res_update = 0.0;
        dtype w_backup[nweights];

        const unsigned int maxiter = 2000;
        unsigned int iter = 0;
        dtype gradient_norm = 1.0;
        string fname;
        fname = "Newton" + to_string(trial) + ".csv";
        csv.open("Exp results_4_10_3/smooth_0.1/Newton/"+fname);

        while ((gradient_norm > 1e-4) && (iter++ < maxiter)) {
                if (iter % 200 == 0) {
                        //cout << "iter " << iter << " gradient norm " << gradient_norm << endl;
                }

                for (unsigned int i = 0; i < nweights; i++){
                    w_backup[i] = dco::value(*w[i]);
                }
                for (unsigned int i = 0; i < nweights; ++i) {
                        gradient[i] = 0.0;
                }
                res = 0.0;
                res_update = 0.0;
                gradient_norm = 0.0;
                for (unsigned int i = 0; i < n; ++i) {
                        ttype eval[3], eval_a[3], eval_m[3], p[3], p_a[3];
                        network->evaluate(0, &x[i * 4], eval);
                        ttype prob[3], prob_a[3];
                        ttype m = eval[0], sum = 0.0, sum_a = 0.0;
                        for (unsigned int j = 1; j < 3; j++){
                            if (m < eval[j]){
                                m = eval[j];
                            }
                        }
                        for (unsigned int j = 0; j < 3; j++){
                            eval_m[j] = eval[j] - m;
                        }
                        for (unsigned int j = 0; j < 3; j++){
                            p[j] = exp(eval_m[j]);
                            sum = sum + p[j];
                        }
                        for (unsigned int j = 0; j < 3; j++){
                            ttype temp = prob[j] = p[j]/sum;
                            prob[j] = max(prob[j], eps);
                            res -= dco::value(y[i * 3 + j] * log(prob[j]));

                            prob_a[j] = -y[i * 3 + j] * (1 / prob[j]) * 1.0;
                            if (temp < eps) {
                                prob_a[j] = 0.0;
                            }
                            p_a[j] = (1 / sum) * prob_a[j];
                            sum_a += -p[j] / (sum * sum) * prob_a[j];
                        }
                        for (int j = 2; j >= 0; j--){
                            p_a[j] += 1.0 * sum_a;
                        }
                        for (unsigned int j = 0; j < 3; j++){
                            eval_a[j] = p[j] * p_a[j];
                        }

                        network->evaluate_adjoint(0, &x[i * 4], eval_a, gradient);
                }
                endt1 = omp_get_wtime();
                csv << iter << "," << res << "," << (endt1 - startt) << endl;
                for (unsigned int i = 0; i < nweights; i++){
                    gradient_norm += dco::value(gradient[i]) * dco::value(gradient[i]);
                }
                gradient_norm = sqrt(gradient_norm);
                MatrixReplacement A;
                A.x_input = x;
                A.y_input = y;
                A.number = n;
                A.lambda = lam;
                A.num_weights = nweights;
                A.test_network = network;
                Eigen::VectorXd b(nweights), delta;
                for (unsigned int i = 0; i < nweights; i++){
                    b(i) = dco::value(gradient[i]);
                }

                Eigen::ConjugateGradient < MatrixReplacement, Eigen::Lower|Eigen::Upper, MyJacobiPreconditioner<dtype> > cg;
                Eigen::VectorXd invdiag(nweights);
                for (unsigned int i = 0; i < nweights; i++){
                    invdiag(i) = 1.0;
                }
                cg.preconditioner().setInvDiag(invdiag);
                cg.compute(A);
                cg.setTolerance(1e-4);
                delta = cg.solve(b);
                if (gradient_norm > 1e-4){
                    for ( int i = 0; i < nweights ; ++i){
                        dco::value(*w[i]) = dco::value(*w[i]) - alpha * delta(i);
                    }
                    for (unsigned int i = 0; i < n; ++i){
                        ttype eval[3], eval_m[3], p[3];
                        network->evaluate(0, &x[i * 4], eval);
                        ttype prob[3];
                        ttype m = eval[0], sum = 0.0 ;
                        for (unsigned int j = 1; j < 3; j++){
                            if (m < eval[j]){
                                m = eval[j];
                            }
                        }
                        for (unsigned int j = 0; j < 3; j++){
                            eval_m[j] = eval[j] - m;
                        }
                        for (unsigned int j = 0; j < 3; j++){
                            p[j] = exp(eval_m[j]);
                            sum = sum + p[j];
                        }
                        for (unsigned int j = 0; j < 3; j++){
                            prob[j] = p[j]/sum;
                            prob[j] = max(prob[j], eps);
                            res_update -= dco::value(y[i * 3 + j] * log(prob[j]));
                        }
                    }
                    if (res_update > res || res_update == res){
                        lam = lam * 10;

                        for (int i = 0; i < nweights; i++) {
                            dco::value(*w[i]) = w_backup[i];
                        }
                    }else{
                        lam = lam * 0.2;
                    }
                }
        }
        csv.close();
        //cout << "lambda is " << lam <<endl;
        cout << "residual of the " << trial << ":" << res << endl;
       // network->print();
        delete [] delta;
        delete [] w;
        delete [] gradient;
}





int main() {
        ofstream csv;
        //dco::ga1s<dtype>::global_tape = dco::ga1s<dtype>::tape_t::create();
        function<ttype(ttype)> sigmoidal = [](ttype x) -> ttype {
                return 1.0 / (1.0 + exp(-x));
                //return tanh(x);
        };

        function<ttype(ttype)> sigmoidal_derivative = [](ttype x) -> ttype {
                return exp(x) / ((exp(x) + 1.0) * (exp(x) + 1.0));
                //return (1.0 - tanh(x) * tanh(x));
        };

        function<ttype(ttype)> rectifier = [](ttype x) -> ttype {
                if (x > 0.0) {
                        return x;
                } else {
                        return 0.0;
                }
        };

        function<ttype(ttype)> rectifier_derivative = [](ttype x) -> ttype {
                if (x > 0.0) {
                        return 1.0;
                } else {
                        return 0.0;
                }
        };

        function<ttype(ttype)> id = [](ttype x) -> ttype {
                return x;
        };

        function<ttype(ttype)> id_derivative = [](ttype x) -> ttype {
                return 1.0;
        };

        function<ttype(ttype)> exponential = [](ttype x) -> ttype{
                return exp(x);
        };
        function<ttype(ttype)> exponential_derivative = [](ttype x) -> ttype{
                return exp(x);
        };
        function<ttype(ttype)> softplus = [](ttype x) -> ttype{
                //return log(1.0 + exp(x));
                return (sqrt(x * x + 4 * epsilon * epsilon) + x)/2;
        };
        function<ttype(ttype)> softplus_derivative = [](ttype x) -> ttype{
                //return 1.0 / (1.0 + exp(-x));
                return x / (2 * sqrt(x * x + 4 * epsilon * epsilon)) + 0.5;
        };

        //ffnn::ffnn_network<dtype> test(1);
        ffnn::ffnn_network<ttype> test(2);
        test.setLayer(0, new ffnn::ffnn_layer<ttype>(nthreads, 4, 10, softplus, softplus_derivative));
        test.setLayer(1, new ffnn::ffnn_layer<ttype>(nthreads, 10, 3, id, id_derivative));
        //test.setLayer(1, new ffnn::ffnn_layer<dtype>(nthreads, 10, 10, sigmoidal, sigmoidal_derivative));
        //test.setLayer(2, new ffnn::ffnn_layer<dtype>(nthreads, 10, 1, id, id_derivative));

        //test.randomWeightsNormal(0.0, 1.0);
        //test.layers[0]->randomWeightsNormal(0.0,sqrt(2.0/4.0));
        //test.layers[1]->randomWeightsNormal(0.0,sqrt(2.0/20.0));
        //test.print();
/*
        // Plot the data
        const int n = 10000;
        dtype dx[n], dy[n];
        generate_data(n, dx, dy);
        csv.open("data.csv");
        for (unsigned int i = 0; i < n; ++i) {
                csv << dx[i] << "," << dy[i] << endl;
        }
        csv.close();

        // Perform regression
        //ffnn_regression(n, dx, dy, 1e-5, test);
        ffnn_regression_gd_ls(n, dx, dy, test);

        // Plot the network function
        dtype x;
        dtype y;
        csv.open("network.csv");
        const dtype xl = -5.0, xu = 5.0, xs = 0.01;
        for (x = xl; x <= xu; x += xs) {
                test.evaluate(0, &x, &y);
                csv << x << "," << y << endl;
        }
        csv.close();
*/
        ifstream file ("iris.data");
        string line;
        vector<double> x;
        vector<string> y;
        while (getline(file, line)) {
            istringstream pair (line);
            string data_1;
            for (unsigned int i = 0; i< 4; i++){
                getline(pair, data_1,',');
                double data_1_d = atof(data_1.c_str());
                x.push_back(data_1_d);
            }
            string data_2;
            getline(pair, data_2); 
            y.push_back(data_2);
        }
        int n = y.size();
        double attr[n * 4], label[n], lab_vec[n * 3];
        vector< vector<double> > lab(n, vector<double>(3,0));
        for (int i = 0; i < n; i++){
            if(y[i] == "Iris-setosa"){
                label[i] = 0;
                lab[i][0] = 1;
            }
            if(y[i] == "Iris-versicolor"){
                label[i] = 1;
                lab[i][1] = 1;
            }
            if(y[i] == "Iris-virginica"){
                label[i] = 2;
                lab[i][2] = 1;
            }
           for (unsigned j = 0; j < 4; j++){
               attr[i * 4 + j] = x[i * 4 + j];
           }

        }
        for (unsigned int i = 0; i < n; i++){
            for (unsigned j = 0; j < 3; j++){
                lab_vec[i * 3 + j] = lab[i][j];
            }
        }
        for (unsigned int i = 0; i < 100; ++i){
            test.layers[0]->randomWeightsNormal(0.0,sqrt(2.0/4.0));
            test.layers[1]->randomWeightsNormal(0.0,sqrt(2.0/10.0));
            ffnn_regression_newton(n, attr, lab_vec, 1.0, &test, 1e-4,i);
        }
        //ffnn_regression_newton(n, attr, lab_vec, 1.0, &test, 1e-4,1);
        double *lab_at = new double[n];
        for (unsigned int i = 0; i < n; i++){
            ttype y_vec[3];
            test.evaluate(0, &attr[i * 4], y_vec);
            lab_at[i] = 0;
            dtype max = dco::value(y_vec[0]);
            for (unsigned int j = 1; j < 3; j++){
                if (dco::value(y_vec[j]) > max){
                    max = dco::value(y_vec[j]);
                    lab_at[i] = j;
                }
            }
        }

        int sum = 0;
        for (unsigned i = 0; i < n; i++){
            if (lab_at[i] == label[i]){
                sum++;
            }
        }

//        dtype acc_eval = (dtype)sum_2/(dtype)num_eval;
        dtype acc_train = (dtype)sum/(dtype)n;
       // dtype acc_test = (dtype)sum_3/(dtype)num_test;
 //       cout << "Accuracy of evaluation: " << acc_eval << endl;
        cout << "Accuracy of training: " << acc_train << endl;
        //cout << "Accuracy of test: " << acc_test << endl;
        //dco::ga1s< dtype >::tape_t::remove(dco::ga1s< dtype >::global_tape);

        return 0;
}



