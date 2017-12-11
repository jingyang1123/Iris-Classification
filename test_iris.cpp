#include <iomanip>
#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <deque>
#include <queue>
using namespace std;

#include <omp.h>
#include "dco_scrambled.hpp"
#include "ffnn_direct_1.h"

#include "Eigen/Dense"
//#include "Eigen/Core"
//#include "Eigen/IterativeLinearSolvers"

using namespace Eigen;
using Eigen::MatrixXd;
#define dtype double
#define ttype dco::gt1s<dtype>::type

const int nthreads = 1;
ofstream csv;

void ffnn_val(const unsigned int n, const dtype* x, const dtype* y, ffnn::ffnn_network<dtype> &network, dtype& res){
        res = 0.0;
        dtype eps = 1e-15;
        for (unsigned int i = 0; i< n; i++){
            dtype eval[3], eval_m[3], p[3];
            network.evaluate(0, &x[i * 4], eval);
            dtype prob[3];
            dtype m = eval[0], sum = 0.0;   
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
                res -= y[i * 3 + j] * log(prob[j]);              
            }

        }       
}

void ffnn_grad(const unsigned int n, const dtype* x, const dtype* y, ffnn::ffnn_network<dtype> &network,  dtype*& grad){

        dtype eps = 1e-15;
        for (unsigned int i = 0; i< n; i++){
            dtype eval[3], eval_a[3], eval_m[3], p[3], p_a[3];
            network.evaluate(0, &x[i * 4], eval);
            dtype prob[3], prob_a[3];
            dtype m = eval[0], sum = 0.0, sum_a = 0.0;

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
                dtype temp = prob[j] = p[j]/sum;
                prob[j] = max(prob[j], eps);
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
            network.evaluate_adjoint(0, &x[i * 4], eval_a, grad);
        }
}

void zoom(const unsigned int n, const dtype* x, const dtype* y, dtype alpha_low, dtype alpha_high, ffnn::ffnn_network<dtype> &network, int nweights, VectorXd direct, dtype c1, dtype c2, dtype& alpha){
        dtype** w = new dtype*[nweights];
        network.getWeights(w);
        dtype* w_backup = new dtype[nweights];
        dtype res, res_new,res_low, wolfe1, wolfe2;
        dtype* grad = new dtype[nweights];

        dtype eps = 1e-15;
        VectorXd gradient_new(nweights), gradient(nweights);

        for (unsigned int i = 0; i < nweights; i++){
            w_backup[i] = *w[i];
        }
        for (unsigned int i = 0; i < nweights; i++){
            grad[i] = 0.0;
        }
        ffnn_val(n,x, y, network, res);

        ffnn_grad(n,x, y, network, grad);
        for (unsigned int i = 0; i < nweights; i++){
            gradient(i) = grad[i];
        }
        wolfe1 = direct.transpose() * gradient;
        while(true){
            dtype alpha_x = 0.5 * (alpha_low + alpha_high);          
            for (unsigned int i = 0; i < nweights; i++){
                *w[i] = w_backup[i];
            }
            for (unsigned int i = 0; i < nweights; i++){
                *w[i] = *w[i] + alpha_low * direct(i);
            }
            ffnn_val(n,x, y, network, res_low);
            for (unsigned int i = 0; i < nweights; i++){
                *w[i] = w_backup[i];
            }
            for (unsigned int i = 0; i < nweights; i++){
                *w[i] = *w[i] + alpha_x * direct(i);
            }
            ffnn_val(n,x, y, network, res_new);
            if (res_new > res + c1 * alpha_x * wolfe1 || res_new >= res_low){
                alpha_high = alpha_x;
            }
            else{
                    dtype* grad_new = new dtype[nweights];
                    ffnn_grad(n,x, y, network, grad_new);
                    for (unsigned int i = 0; i < nweights; i++){
                        gradient_new(i) = grad_new[i];
                    }
                    wolfe2 = direct.transpose() * gradient_new;
                    if(abs(wolfe2) <= -c2 * wolfe1 ){
                        alpha = alpha_x;
                        break;
                    }
                    if(wolfe2 * (alpha_high - alpha_low) >= 0.0 ){
                        alpha_high = alpha_low;
                    }
                    alpha_low = alpha_x;
            }

            if(abs(alpha_low - alpha_high) < eps){
                alpha = alpha_x;
                break;
            }

        }

}

void line_search(const unsigned int n, const dtype* x, const dtype* y, ffnn::ffnn_network<dtype> &network, int nweights, VectorXd direct, dtype alpha_max, dtype c1, dtype c2, dtype& alpha ){
        dtype** w = new dtype*[nweights];
        network.getWeights(w);
        dtype* w_backup = new dtype[nweights];
        dtype res, res_new, wolfe1, wolfe2, alpha_old;
        dtype* grad = new dtype[nweights];

        VectorXd gradient_new(nweights), gradient(nweights);

        dtype alpha_it = alpha_max * 0.5;
        for (unsigned int i = 0; i < nweights; i++){
            w_backup[i] = *w[i];
        }
        for (unsigned int i = 0; i < nweights; i++){
            grad[i] = 0.0;
        }
        ffnn_val(n, x, y, network, res);
        ffnn_grad(n,x, y, network, grad);
        for (unsigned int i = 0; i < nweights; i++){
            gradient(i) = grad[i];
        }
        wolfe1 = direct.transpose() * gradient;
        int t = 1;
        dtype res_old = res;
        alpha_old = 0.0;
        while( t > 0){
            for (unsigned int i = 0; i < nweights; i++){
                *w[i] = w_backup[i];
            }
            for (unsigned int i = 0; i < nweights; i++){
                *w[i] = *w[i] + alpha_it * direct(i);               
            }
            ffnn_val(n, x, y, network, res_new);
            if(res_new > res + c1 * alpha_it * wolfe1 || ((res_new >= res_old) && (t > 1))){
                for (unsigned int i = 0; i < nweights; i++){
                    *w[i] = w_backup[i];
                }
                zoom(n,x,y,alpha_old,alpha_it,network, nweights, direct,c1,c2,alpha);
                break;
           }
            dtype* grad_new = new dtype[nweights];
            ffnn_grad(n,x, y, network, grad_new);
            for (unsigned int i = 0; i < nweights; i++){
                gradient_new(i) = grad_new[i];
            }
            wolfe2 = direct.transpose() * gradient_new;
            if(abs(wolfe2) <= -c2 * wolfe1){
                alpha = alpha_it;
                break;
            }
            if(wolfe2 >= 0){
                for (unsigned int i = 0; i < nweights; i++){
                    *w[i] = w_backup[i];
                }
                zoom(n,x,y,alpha_it,alpha_old,network, nweights, direct,c1,c2,alpha);
                break;
            }
            res_old = res_new;
            alpha_old = alpha_it;
            alpha_it = 0.5 * (alpha_it + alpha_max);
            t = t + 1;
       }

}

void inexact_line_search(const unsigned int n, const dtype* x, const dtype* y, ffnn::ffnn_network<dtype> &network, int nweights, VectorXd direct, dtype c1, dtype c2, dtype& alpha){
    dtype** w = new dtype*[nweights];
    network.getWeights(w);
    dtype* w_backup = new dtype[nweights];
    dtype res, res_new, wolfe1, wolfe2, alpha_old;
    dtype* grad = new dtype[nweights];

    VectorXd gradient_new(nweights), gradient(nweights);

    dtype alpha_it = 1.0;
    dtype alpha_min = 0;
    dtype alpha_max = 1e15;

    for (unsigned int i = 0; i < nweights; i++){
        w_backup[i] = *w[i];
    }
    for (unsigned int i = 0; i < nweights; i++){
        grad[i] = 0.0;
    }
    ffnn_val(n, x, y, network, res);
    ffnn_grad(n,x, y, network, grad);
    for (unsigned int i = 0; i < nweights; i++){
        gradient(i) = grad[i];
    }
    wolfe1 = direct.transpose() * gradient;
    alpha_old = 0.0;
    while( true){
        for (unsigned int i = 0; i < nweights; i++){
            *w[i] = w_backup[i];
        }
        for (unsigned int i = 0; i < nweights; i++){
            *w[i] = *w[i] + alpha_it * direct(i);
        }
        ffnn_val(n, x, y, network, res_new);
        if(res_new > res + c1 * alpha_it * wolfe1){
            alpha_max = alpha_it;
        }
        else{
            dtype* grad_new = new dtype[nweights];
            ffnn_grad(n,x, y, network, grad_new);
            for (unsigned int i = 0; i < nweights; i++){
                gradient_new(i) = grad_new[i];
            }
            wolfe2 = direct.transpose() * gradient_new;
            if(wolfe2 <= c2 * wolfe1){
                alpha_min = alpha_it;
            }
            else{
                alpha = alpha_it;
                break;
            }
        }
        if(alpha_max < 1e15){
            alpha_it = (alpha_min + alpha_max)*0.5;
        }
        else{
            alpha_it =2 * alpha_min;
        }

    }
}

void ffnn_regression_gd(const unsigned int n, const dtype* x, const dtype* y, dtype alpha, ffnn::ffnn_network<dtype> &network, const unsigned int trial) {
        double startt, endt, endt1;
        startt = omp_get_wtime();

        unsigned int nweights = network.getNWeights();
        dtype** w = new dtype*[nweights];
        network.getWeights(w);
        dtype* gradient = new dtype[nweights];

        const unsigned int maxiter = 10000;
        unsigned int iter = 0;
        dtype gradient_norm = 1.0;
        dtype res = 0.0;
        string fname;
        //fname = "gd" + to_string(trial) + ".csv";
        //csv.open("Exp results_4_30_3/smooth_0.1/GD/"+fname);
        ffnn_val(n,x, y, network, res);
        //endt1 = omp_get_wtime();
        //csv << iter << "," << res << "," << (endt1 - startt) << endl;

        while ((gradient_norm > 1e-4) && (iter++ < maxiter)) {
                if (iter % 5000 == 0) {
                        //cout << "iter " << iter << " gradient_norm " << gradient_norm << endl;
                }

                for (unsigned int j = 0; j < nweights; ++j) {
                        gradient[j] = 0.0;
                }

                ffnn_grad(n,x, y, network, gradient);
                gradient_norm = 0.0;
                for (unsigned int i = 0; i < nweights; ++i) {
                        *w[i] = *w[i] - alpha * gradient[i];
                        gradient_norm += gradient[i] * gradient[i];
                }
                gradient_norm = sqrt(gradient_norm);
                ffnn_val(n,x, y, network, res);
               // endt = omp_get_wtime();
                //csv << iter << "," << res << "," << (endt - startt) << endl;
                //cout << "gradient norm: " << gradient_norm << endl;
        }
        //csv.close();
        //cout << "iter " << iter << " gradient_norm " << gradient_norm << endl;
        cout << "residual of gd at " << trial << ":" << res << endl;

       // network.print();

        delete [] gradient;
        delete [] w;
}



void ffnn_regression_gd_ls(const unsigned int n, const dtype* x, const dtype* y, ffnn::ffnn_network<dtype> &network, const unsigned int trial) {
        double startt, endt, endt1;
        startt = omp_get_wtime();

        unsigned int nweights = network.getNWeights();
        dtype** w = new dtype*[nweights];
        network.getWeights(w);
        dtype* gradient = new dtype[nweights];
        dtype wolfe_val;
        dtype c1 = 1e-4;
        dtype w_backup[nweights];
        dtype res_new = 0.0;
        dtype res = 0.0;
        dtype eps = 1e-15;

        const unsigned int maxiter = 10000;
        unsigned int iter = 0;
        dtype gradient_norm = 1.0;

        string fname;
        //fname = "gd_ls" + to_string(trial) + ".csv";
        //csv.open("Exp results_4_30_3/smooth_0.1/GD_ls/"+fname);
        ffnn_val(n,x, y, network, res);
        //endt1 = omp_get_wtime();
        //csv << iter << "," << res << "," << (endt1 - startt) << endl;

        while ((gradient_norm > 1e-4) && (iter++ < maxiter)) {
                if (iter % 2000 == 0) {
                        //cout << "iter " << iter << " gradient norm " << gradient_norm << endl;
                }
                dtype alpha = 1.0;

                res_new = 0.0;

                for (unsigned int i = 0; i < nweights; i++){
                    w_backup[i] = *w[i];
                    //cout << "weights "<< i << ": "<< w_backup[i] << " ";
                }//cout << endl;

                for (unsigned int i = 0; i < nweights; ++i) {
                        gradient[i] = 0.0;
                }
                ffnn_grad(n,x, y, network, gradient);

                //cout << "res: " << res << endl;
                gradient_norm = 0.0;
                for (unsigned int i = 0; i < nweights; ++i) {
                        *w[i] = *w[i] - alpha * gradient[i];
                        gradient_norm += gradient[i] * gradient[i];
                        //cout << "gradient" << i <<":" << gradient[i] <<endl;
                }

                wolfe_val = -gradient_norm;
                ffnn_val(n,x, y, network, res_new);

                //cout << "res_new: " << res_new << endl;
                while (res_new > res + c1 * alpha * wolfe_val){
                    alpha = alpha * 0.5;
                    res_new = 0.0;

                    //cout << "dis_step " << dis_step << endl;

                    for (unsigned int i = 0; i < nweights ; ++i){

                        *w[i] = w_backup[i];

                        *w[i] = *w[i] - alpha * gradient[i];


                    }//cout << endl;

                    ffnn_val(n,x, y, network, res_new);
                }
                //endt = omp_get_wtime();
                //csv << iter << "," << res_new << "," << (endt - startt) << endl;
                gradient_norm = sqrt(gradient_norm);
                res = res_new;
        }
        csv.close();

        //cout << "iter " << iter << " gradient norm " << gradient_norm << endl;
        cout << "residual of gd_ls at " << trial << ":" << res_new << endl;

        //network.print();

        delete [] gradient;
        delete [] w;
}

void ffnn_regression_BFGS(const unsigned int n, const dtype* x, const dtype* y, ffnn::ffnn_network<dtype> &network, const unsigned int trial){
        double startt, endt, endt1;
        startt = omp_get_wtime();


        unsigned int nweights = network.getNWeights();
        dtype** w = new dtype*[nweights];
        network.getWeights(w);
        dtype *grad_new = new dtype[nweights];
        dtype *grad_update = new dtype[nweights];
        VectorXd direct(nweights);
        dtype gradient_norm = 1.0;
        dtype wolfe_val_1 = 0.0;
        dtype wolfe_val_2 = 0.0;
        dtype c1 = 1e-4;
        dtype c2 = 0.9;
        dtype eps = 1e-307;
        unsigned int m = 6;
        deque <VectorXd> s_k, y_k;
        deque <double> rho;

        MatrixXd Inv_hessian = MatrixXd::Identity(nweights,nweights);

        dtype res_new = 0.0, res_old;

        for (unsigned int i = 0; i < nweights; i++){
            grad_new[i] = 0.0;
        }

        ffnn_grad(n, x, y, network, grad_new);

        VectorXd gradient_new(nweights), gradient(nweights);
        for (unsigned i = 0; i < nweights; i++){
            gradient_new(i) = grad_new[i];
        }
        //cout << "res: " << res_new << endl;
        //ffnn_val_grad(n, x, y, network,  res_old,  grad_update);
        //cout << "res2: " << res_old << endl;
/*
        for (unsigned int i = 0; i < nweights; i++){
            cout << "weight:" << *w[i] << ",";
        }cout << endl;
*/

        const unsigned int maxiter = 2000;
        unsigned int iter = 0;

        string fname;
        fname = "BFGS_wolfe" + to_string(trial) + ".csv";
        csv.open("Exp results_4_30_3/smooth_0.1/BFGS_wolfe/"+fname);
        ffnn_val(n,x, y, network, res_old);
        endt1 = omp_get_wtime();
        csv << iter << "," << res_old << "," << (endt1 - startt) << endl;

        while((gradient_norm > 1e-4) && (iter++ < maxiter)){
            if (iter % 20 == 0) {
                    //cout << "iter " << iter << " gradient norm " << gradient_norm << endl;
            }
            gradient_norm = 0.0;
            gradient = gradient_new;
            direct =  -Inv_hessian * gradient;
            dtype alpha;
            line_search(n, x, y, network, nweights, direct, 2.0,  c1, c2, alpha );
/*
            for ( unsigned int i =0; i < nweights; i++){
                cout << "weights in fc pr: "<< *w[i] << ",";
            }cout << endl;

            for (unsigned int i = 0; i < nweights; i++){
                *w[i] = *w[i] + alpha * direct(i);
            }
            for ( unsigned int i =0; i < nweights; i++){
                cout << "weights in fc: "<< *w[i] << ",";
            }cout << endl;
*/
            for (unsigned int i = 0; i < nweights; i++){
                grad_update[i] = 0.0;
            }
            ffnn_grad(n, x, y, network, grad_update);
            ffnn_val(n, x, y, network, res_new);
            endt = omp_get_wtime();
            csv << iter << "," << res_new << "," << (endt - startt) << endl;
            VectorXd dis_step(nweights);
            dis_step = alpha * direct;
            for (unsigned i = 0; i < nweights; i++){
                gradient_new(i) = grad_update[i];
            }
            for (unsigned int i = 0; i < nweights; i++){
                gradient_norm += gradient_new(i) * gradient_new(i);
            }
            gradient_norm = sqrt(gradient_norm);
            VectorXd delta_grad(nweights);
            delta_grad = gradient_new - gradient;
            if (delta_grad.norm() < eps && gradient_norm > 1e-4){
                cout << trial <<endl;
                break;
            }
            double deno;
            deno = 1 /(dis_step.transpose() * delta_grad);
            MatrixXd mat1(nweights,nweights), mat2(nweights,nweights);
            mat1 = delta_grad * dis_step.transpose() * deno;
            mat2 = dis_step * dis_step.transpose() * deno;
            MatrixXd iden = MatrixXd::Identity(nweights,nweights);
            MatrixXd mat3 = iden - mat1;
            Inv_hessian = mat3.transpose() * Inv_hessian * mat3 + mat2;
        }

        csv.close();
        delete [] w;

}
void ffnn_regression_BFGS_nonsmooth(const unsigned int n, const dtype* x, const dtype* y, ffnn::ffnn_network<dtype> &network, const unsigned int trial){
        double startt, endt, endt1;
        startt = omp_get_wtime();


        unsigned int nweights = network.getNWeights();
        dtype** w = new dtype*[nweights];
        network.getWeights(w);
        dtype *grad_new = new dtype[nweights];
        dtype *grad_update = new dtype[nweights];
        VectorXd direct(nweights);
        dtype gradient_norm = 1.0;
        dtype wolfe_val_1 = 0.0;
        dtype wolfe_val_2 = 0.0;
        dtype c1 = 1e-4;
        dtype c2 = 0.9;
        dtype eps = 1e-15;
        unsigned int m = 6;
        deque <VectorXd> s_k, y_k;
        deque <double> rho;

        MatrixXd Inv_hessian = MatrixXd::Identity(nweights,nweights);

        dtype res_new = 0.0, res_old;

        for (unsigned int i = 0; i < nweights; i++){
            grad_new[i] = 0.0;
        }

        ffnn_grad(n, x, y, network, grad_new);

        VectorXd gradient_new(nweights), gradient(nweights);
        for (unsigned i = 0; i < nweights; i++){
            gradient_new(i) = grad_new[i];
        }
/*
        for (unsigned int i = 0; i < nweights; i++){
            cout << "weight:" << *w[i] << ",";
        }cout << endl;
*/

        const unsigned int maxiter = 2000;
        unsigned int iter = 0;

        string fname;
        fname = "BFGS_wolfe" + to_string(trial) + ".csv";
        csv.open("Exp results_4_10_3/rectifier_2/BFGS_wolfe/"+fname);
        //csv.open("Test/"+fname);
        ffnn_val(n,x, y, network, res_old);
        endt1 = omp_get_wtime();
        csv << iter << "," << res_old << "," << (endt1 - startt) << endl;

        while((gradient_norm > 1e-4) && (iter++ < maxiter)){
            if (iter % 20 == 0) {
                    //cout << "iter " << iter << " gradient norm " << gradient_norm << endl;
            }
            gradient_norm = 0.0;
            gradient = gradient_new;
            direct =  -Inv_hessian * gradient;
            dtype alpha;
            inexact_line_search(n, x, y, network, nweights, direct, c1, c2, alpha );
/*
            for ( unsigned int i =0; i < nweights; i++){
                cout << "weights in fc pr: "<< *w[i] << ",";
            }cout << endl;

            for (unsigned int i = 0; i < nweights; i++){
                *w[i] = *w[i] + alpha * direct(i);
            }
            for ( unsigned int i =0; i < nweights; i++){
                cout << "weights in fc: "<< *w[i] << ",";
            }cout << endl;
*/
            //cout << "REALLY THIS" << endl;
            for (unsigned int i = 0; i < nweights; i++){
                grad_update[i] = 0.0;
            }
            ffnn_grad(n, x, y, network, grad_update);
            ffnn_val(n, x, y, network, res_new);
            endt = omp_get_wtime();
            csv << iter << "," << res_new << "," << (endt - startt) << endl;
            VectorXd dis_step(nweights);
            dis_step = alpha * direct;
            for (unsigned i = 0; i < nweights; i++){
                gradient_new(i) = grad_update[i];
            }
            VectorXd delta_grad(nweights);
            delta_grad = gradient_new - gradient;
            if (delta_grad.norm() < eps){
                cout << trial <<endl;
                break;
            }
            double deno;
            deno = 1 /(dis_step.transpose() * delta_grad);
            MatrixXd mat1(nweights,nweights), mat2(nweights,nweights);
            mat1 = delta_grad * dis_step.transpose() * deno;
            mat2 = dis_step * dis_step.transpose() * deno;
            MatrixXd iden = MatrixXd::Identity(nweights,nweights);
            MatrixXd mat3 = iden - mat1;
            Inv_hessian = mat3.transpose() * Inv_hessian * mat3 + mat2;
            for (unsigned int i = 0; i < nweights; i++){
                gradient_norm += gradient_new(i) * gradient_new(i);
            }
            gradient_norm = sqrt(gradient_norm);
        }

        csv.close();
        cout << "residual of BFGS at " << trial << ":" << res_new << endl;
        delete [] w;

}

void ffnn_regression_BFGS_bt(const unsigned int n, const dtype* x, const dtype* y, ffnn::ffnn_network<dtype> &network){

        unsigned int nweights = network.getNWeights();
        dtype** w = new dtype*[nweights];
        network.getWeights(w);
        dtype *grad_new = new dtype[nweights];
        dtype *grad_update = new dtype[nweights];
        VectorXd direct(nweights);
        dtype gradient_norm = 1.0;
        dtype wolfe_val_1 = 0.0;
        dtype c1 = 1e-4;
        dtype c2 = 0.9;
        dtype eps = 1e-15;
        unsigned int m = 6;

        MatrixXd Inv_hessian = MatrixXd::Identity(nweights,nweights);

        dtype res_new = 0.0, res_old;

        for (unsigned int i = 0; i < nweights; i++){
            grad_new[i] = 0.0;
        }

        ffnn_val(n, x, y, network,  res_new);
        ffnn_grad(n, x, y, network,  grad_new);
        VectorXd gradient_new(nweights), gradient(nweights);
        for (unsigned i = 0; i < nweights; i++){
            gradient_new(i) = grad_new[i];
        }
        double w_backup[nweights];
        const unsigned int maxiter = 2000;
        unsigned int iter = 0;
        csv.open("BFGS_bt.csv");
        csv << iter << "," << res_new << endl;
        while((gradient_norm > 1e-4) && (iter++ < maxiter)){
            if (iter % 20 == 0) {
                    cout << "iter " << iter << " gradient norm " << gradient_norm << endl;
            }
            double alpha = 1.0;
            res_old = res_new;
            res_new = 0.0;
            gradient_norm = 0.0;
            gradient = gradient_new;
            direct =  -Inv_hessian * gradient;
            wolfe_val_1 = gradient.transpose()* direct;
            for (unsigned int i = 0; i < nweights; i++){
                w_backup[i] = *w[i];
            }
            for (unsigned int i = 0; i < nweights ; ++i){

                *w[i] = *w[i] + alpha * direct(i);
            }

            ffnn_val(n, x, y, network,  res_new);
            VectorXd dis_step(nweights);
            // line search
            while ((res_new > res_old + c1 * alpha * wolfe_val_1)){
                alpha = alpha * 0.5;
                res_new = 0.0;

                dis_step = alpha * direct;
                for (unsigned int i = 0; i < nweights ; ++i){
                    *w[i] = w_backup[i];
                    *w[i] = *w[i] + dis_step(i);
                }
                ffnn_val(n, x, y, network,  res_new);
            }
            csv << iter << "," << res_new << endl;
            for (unsigned int i = 0; i < nweights; i++){
                grad_update[i] = 0.0;
            }
            ffnn_grad(n, x, y, network,  grad_update);
            for (unsigned i = 0; i < nweights; i++){
                gradient_new(i) = grad_update[i];
            }
            VectorXd delta_grad(nweights);
            delta_grad = gradient_new - gradient;
/*
            if (delta_grad.norm() < eps){
                break;
            }
*/
            double deno;
            deno = (dis_step.transpose() * delta_grad);
            if( deno < eps){
                break;
            }
            deno = 1.0 / deno;

            MatrixXd mat1(nweights,nweights), mat2(nweights,nweights);
            mat1 = delta_grad * dis_step.transpose() * deno;
            mat2 = dis_step * dis_step.transpose() * deno;
            MatrixXd iden = MatrixXd::Identity(nweights,nweights);
            MatrixXd mat3 = iden - mat1;
            Inv_hessian = mat3.transpose() * Inv_hessian * mat3 + mat2;
            for (unsigned int i = 0; i < nweights; i++){
                gradient_norm += gradient_new(i) * gradient_new(i);
            }
            gradient_norm = sqrt(gradient_norm);
        }

        csv.close();
        cout << "iter " << iter << " gradient norm " << gradient_norm << endl;
        cout << "residual: " << res_new << endl;
        delete [] w;
}


int main() {
        ofstream csv;

        cout << setprecision(16);

        function<dtype(dtype)> sigmoidal = [](dtype x) -> dtype {
                return 1.0 / (1.0 + exp(-x));
                //return tanh(x);
        };

        function<dtype(dtype)> sigmoidal_derivative = [](dtype x) -> dtype {
                return (1.0/(1.0 + exp(-x))) * (1.0 - (1.0/(1.0 + exp(-x))));
                //return (1.0 - tanh(x) * tanh(x));
        };

        function<dtype(dtype)> rectifier = [](dtype x) -> dtype {
                if (x > 0.0) {
                        return x;
                } else {
                        return 0.0;
                }
        };

        function<dtype(dtype)> rectifier_derivative = [](dtype x) -> dtype {
                if (x > 0.0) {
                        return 1.0;
                } else {
                        return 0.0;
                }
        };

        function<dtype(dtype)> id = [](dtype x) -> dtype {
                return x;
        };

        function<dtype(dtype)> id_derivative = [](dtype x) -> dtype {
                return 1.0;
        };

        function<dtype(dtype)> exponential = [](dtype x) -> dtype{
                //cout << "x " << x << " exp(x) " << exp(x) << endl;
                return exp(x);
        };
        function<dtype(dtype)> exponential_derivative = [](dtype x) -> dtype{
                return exp(x);
        };
        function<dtype(dtype)> softplus = [](dtype x) -> dtype{
                //return log(1.0 + exp(x));
                return (sqrt(x * x + 4 * 0.1 * 0.1) + x)/2;
        };
        function<dtype(dtype)> softplus_derivative = [](dtype x) -> dtype{
                //return 1.0 / (1.0 + exp(-x));
                return x / (2 * sqrt(x * x + 4 * 0.1 * 0.1)) + 0.5;
        };

        ffnn::ffnn_network<dtype> test(2);

        //test.setLayer(0, new ffnn::ffnn_layer<dtype>(nthreads, 784, 300, rectifier, rectifier_derivative));

        test.setLayer(0, new ffnn::ffnn_layer<dtype>(nthreads, 4, 30, softplus, softplus_derivative));

        test.setLayer(1, new ffnn::ffnn_layer<dtype>(nthreads, 30, 3, id, id_derivative));

        //test.setLayer(2, new ffnn::ffnn_layer<dtype>(nthreads, 10, 1, id, id_derivative));

        //test.randomWeightsNormal(0.0, 1.0);
        //test.layers[0]->randomWeightsNormal(0.0,sqrt(2.0/4.0));
        //test.layers[1]->randomWeightsNormal(0.0,sqrt(2.0/20.0));
        //test.layers[0]->randomWeightsNormal(0.0,1.0);
        //test.layers[1]->randomWeightsNormal(0.0,1.0);
        //test.print();

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

            //double data_2_d = atof(data_2.c_str());

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
            test.layers[1]->randomWeightsNormal(0.0,sqrt(2.0/30.0));
/*
            unsigned int nweights = test.getNWeights();
            dtype** w = new dtype*[nweights];
            dtype* w_backup = new dtype[nweights];
            test.getWeights(w);
            for (unsigned j = 0; j < nweights; ++j){
                w_backup[j] = *w[j];
            }
*/
            ffnn_regression_gd(n, attr, lab_vec, 1e-3, test, i);
/*
            for (unsigned j = 0; j < nweights; ++j){
                *w[j] = w_backup[j];
            }

            ffnn_regression_gd_ls(n, attr, lab_vec, test, i);
/*
            for (unsigned j = 0; j < nweights; ++j){
                *w[j] = w_backup[j];
            }

            //ffnn_regression_BFGS_nonsmooth(n, attr, lab_vec, test, i);
            ffnn_regression_BFGS(n, attr, lab_vec, test, i);
*/
        }

        //ffnn_regression_gd(n, attr, lab_vec, 1e-4, test,1);
        //ffnn_regression_gd_ls(n, attr, lab_vec, test);
        //ffnn_regression_BFGS(n, attr, lab_vec, test,1);

        //ffnn_regression_BFGS_bt(n, attr, lab_vec, test);
        //ffnn_regression_BFGS_nonsmooth(n, attr, lab_vec, test,1);

/*
        dtype *img_eval = new dtype[num_eval*784];
        int *lab_eval = new int[num_eval];
        for (unsigned int i = 0; i < num_eval; i++){
            lab_eval[i] = lab[num_train + i];
            for (unsigned int j = 0; j < size_imag; j++){
                img_eval[i * size_imag + j] = imag[(i + num_train) * size_imag + j];
            }
        }

        int *lab_nw = new int[num_eval];
        for (unsigned int i = 0; i < num_eval; i++){
            dtype y_vec[10];
            test.evaluate(0, &img_eval[i * 784], y_vec);
            lab_nw[i] = 0;
            dtype max = abs(y_vec[0]);
            for (unsigned int j = 1; j < 10; j++){
                if (abs(y_vec[j]) > max){
                    max = abs(y_vec[j]);
                    lab_nw[i] = j;
                }
            }
        }
*/
        double *lab_at = new double[n];
        for (unsigned int i = 0; i < n; i++){
            dtype y_vec[3];
            test.evaluate(0, &attr[i * 4], y_vec);
            lab_at[i] = 0;
            dtype max = y_vec[0];
            for (unsigned int j = 1; j < 3; j++){
                if (y_vec[j] > max){
                    max = y_vec[j];
                    lab_at[i] = j;
                }
            }
        }
/*
        int *lab_ts = new int[num_test];
        for (unsigned int i = 0; i < num_test; i++){
            dtype y_vec[10];
            test.evaluate(0, &imag_test[i * 784], y_vec);
            lab_ts[i] = 0;
            dtype max = abs(y_vec[0]);
            for (unsigned int j = 1; j < 10; j++){
                if (abs(y_vec[j]) > max){
                    max = abs(y_vec[j]);
                    lab_ts[i] = j;
                }
            }
        }
*/
        int sum = 0;
        for (unsigned i = 0; i < n; i++){
            if (lab_at[i] == label[i]){
                sum++;
            }
        }
/*
        int sum_2 = 0;
        for (unsigned i = 0; i < num_eval; i++){
            if (lab_eval[i] == lab_nw[i]){
                sum_2++;
            }
        }

        int sum_3 = 0;
        for (unsigned i = 0; i < num_test; i++){
            if (lab_ts[i] == lab_test[i]){
                sum_3++;
            }
        }
*/
//        dtype acc_eval = (dtype)sum_2/(dtype)num_eval;
        dtype acc_train = (dtype)sum/(dtype)n;
//        dtype acc_test = (dtype)sum_3/(dtype)num_test;
//        cout << "Accuracy of evaluation: " << acc_eval << endl;
        cout << "Accuracy of training: " << acc_train << endl;
       // cout << "Accuracy of test: " << acc_test << endl;

        return 0;
}



