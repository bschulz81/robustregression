/*

Copyright(c) < 2023 > <Benjamin Schulz>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/




#pragma once
#include <valarray>
#include <functional>
#include "matrixcode.h"
#include "lossfunctions.h"
#include "robustregressionlib_exports.h"
using namespace std;

namespace  Non_Linear_Regression
{

	typedef function<valarray<double>(const valarray<double>& X, valarray<double>& beta)> fun;
	typedef function<Matrix(const valarray<double>& X, valarray<double>& beta)> Jacobian;



	struct initdata
	{
		valarray<double> initialguess;
		fun f{ NULL };
		Jacobian J{ NULL };
	};

	//the initial data for nonlinear regression. 
	//initialguess is a first guess for the regression parameters beta. beta is a vector in form of a valarray
	// f(X,beta) is the function whose parameters beta are to be found. X is a valarray describing the x axis, beta are the parameters
	// of the function in form of a valarray. For example, a line could be described as the following function

	/*
	valarray<double> linear(const valarray<double>& X, const  valarray<double>& beta)
	{
		valarray<double> Y(X.size());
		for (size_t i = 0; i < X.size(); i++)
			Y[i] = beta[0] * X[i] + beta[1];
		return Y;
	}
	*/

	// J is the Jacobian Matrix of f(X,beta). For example for the above line, a jacobian matrix would be

	/*
		Matrix Jacobi(const valarray<double>& X, const  valarray<double>& beta)
		{
			Matrix ret(X.size(), beta.size());
			for (size_t i = 0; i < X.size(); i++)
			{
				ret(i, 0) = X[i];
				ret(i, 1) = 1;
			}
			return ret;
		}

	*/







	struct result
	{
		valarray<double> beta;
	};
	//the found parameters as result.



	struct control :LossFunctions::errorfunction
	{
		double lambda{ 4.0 };
		double increment{ 1.5 };
		double decrement{ 5.0 };
		double h{ 0.1 };
		double precision{ 0.000001 };
		double tolerable_error{ 0.000001 };
		size_t stop_nonlinear_curve_fitting_after_iterations{ 40 };
		double stop_nonlinear_curve_fitting_after_seconds{ 30.0 };
	};
	// parameters of the Levenberg-Marquardt algorithm. 
	// with most parameter names according to  https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
	// Usually, the default parameters for lambda/increment/decrement and h work
	// if more precision is desired, you may change the values precision (which is a measure of the precision the increments of the result)
	// and tolerable_error, which measures the error between the fit and the data.
	// but you can also stop the algorithm after some number of iterations or some seconds.


	//Computes the non linear regression from given arrays X,Y, initdata with a function to be fit and a jacobian, given controldata and yields the result
	ROBUSTREGRESSION_API bool non_linear_regression(const valarray<double>& X, const valarray<double>& Y,
		initdata& init,
		control& control,
		result& res);

	//Computes the loss function given the  loss function given the function f(X,beta), the arrays for X and beta, the data Y and the error metric.
	//the output is stored in res
	ROBUSTREGRESSION_API double nonlinear_loss_function(fun f, const valarray<double>& X, valarray<double>& beta, const valarray<double>& Y,
		Non_Linear_Regression::control& ctrl,
		LossFunctions::error& res);
}
