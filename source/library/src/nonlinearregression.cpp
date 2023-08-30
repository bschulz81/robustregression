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

#include <chrono>


#include "statisticfunctions.h"
#include "matrixcode.h"
#include "lossfunctions.h"
#include "nonlinearregression.h"


valarray<double> directionalderivative(Non_Linear_Regression::fun f, const valarray<double>& X, valarray<double>& fxbeta, valarray<double>& beta, valarray<double>& delta, Matrix & J, double h);


inline valarray<double> directionalderivative(Non_Linear_Regression::fun f, const valarray<double>& X, valarray<double>& fxbeta, valarray<double>& beta, valarray<double>& delta, Matrix& J, double h)
{
	return  ((f(X, (beta + (delta * h))) - fxbeta) / h - J * delta) * 2 / h;
}

ROBUSTREGRESSION_API inline bool Non_Linear_Regression::non_linear_regression(const valarray<double>& x,const valarray<double>& y,
	 Non_Linear_Regression::initdata &init,
	Non_Linear_Regression::control &control,
	Non_Linear_Regression::result& res)

{
	if (init.f== NULL)
		return false;
	if (init.J == NULL)
		return false;
	valarray<double> beta = init.initialguess;
	valarray<double> delta1(beta.size());
	valarray<double> delta2(beta.size());
	valarray<double> delta1a(beta.size());
	valarray<double> delta2a(beta.size());
	valarray<double> delta1b(beta.size());
	valarray<double> delta2b(beta.size());

	double err = DBL_MAX;
	LossFunctions::error errs;
	err=nonlinear_loss_function(init.f, x,beta, y, control, errs);
	res.beta = beta;

	double seconds = 0.0;

	auto start = std::chrono::steady_clock::now();

	Jacobian Jac;
	
	Jac = init.J;

	size_t count = 0;

	Matrix J = Jac(x, beta);

	if (J.Rows() != x.size() || J.Columns() != beta.size())
		return false;


	do
	{
		J = Jac(x, beta);
		
		Matrix Jt = Matrixcode::Transpose(J);

		valarray<double> fxbeta = init.f(x, beta);

		valarray<double>  v = Jt * (y - fxbeta);

		Matrix G = Jt * J;

		delta1a=Matrixcode::Gaussian_algorithm(((G)+(Matrixcode::Diagonal(G) * control.lambda)), v);

		double  norm1a = (delta1a * delta1a).sum();
		if (isnan(norm1a))
		{
			break;
		}

		delta1 = delta1a;

		valarray<double> v1b = Jt * directionalderivative(init.f,x, fxbeta, beta, delta1a, J,control.h);

		delta1b = Matrixcode::Gaussian_algorithm(((G)+(Matrixcode::Diagonal(G) * control.lambda)), ((v1b) * (-0.5)));


		double  norm1b = (delta1b * delta1b).sum();
		if ((2.0 * sqrt(norm1b) / sqrt(norm1a)) < 0.1)
		{
			delta1 = delta1 + delta1b;
		}
		norm1a = (delta1a * delta1a).sum();


		valarray<double> temp = beta + delta1;

		LossFunctions::error s1 {};


		nonlinear_loss_function(init.f, x, temp, y,control,s1);

		delta2a=Matrixcode::Gaussian_algorithm(((G)+(Matrixcode::Diagonal(G) * control.lambda / control.decrement)), v);


		double  norm2a = (delta2a * delta2a).sum();

		if (isnan(norm2a))
		{
			break;
		}

		delta2 = delta2a;

		valarray<double> v2b = Jt * directionalderivative(init.f,x, fxbeta, beta, delta2a, J,control.h);

		delta2b = Matrixcode::Gaussian_algorithm(((G)+(Matrixcode::Diagonal(G) * control.lambda / control.decrement)), ((v2b) * (-0.5)));


		double  norm2b = (delta2b * delta2b).sum();
		if ((2.0 * sqrt(norm2b) / sqrt(norm2a)) < 0.1)
		{
			delta2 = delta2 + delta2b;
		}

		norm2a = (delta2a * delta2a).sum();


		valarray<double> temp2 = beta + delta2;

		LossFunctions::error s2 {};

		nonlinear_loss_function(init.f, x, temp2, y, control,s2);


		if (norm2a < control.precision)
		{
			break;
		}
		if (norm1a < control.precision)
		{
			break;
		}
		
		if ((s1.main_error > err) && (s2.main_error >err))
		{
			control.lambda *= control.increment;
			continue;
		}

		if (s2.main_error < s1.main_error)
		{
			control.lambda = control.lambda / control.decrement;
			beta = temp2;
			err = s2.main_error;
			res.beta = beta;
		}
		else
		{
			beta = temp;
			err = s1.main_error;
			res.beta = beta;
		}

		if (err < control.tolerable_error)
			break;

		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		seconds = elapsed_seconds.count();

		count++;
		if (count > control.stop_nonlinear_curve_fitting_after_iterations)
			break;

	} while (seconds < control.stop_nonlinear_curve_fitting_after_seconds);

	res.beta = beta;

	return true;
}

ROBUSTREGRESSION_API inline double Non_Linear_Regression::nonlinear_loss_function(fun f, const valarray<double>& X, valarray<double>& beta, const valarray<double>& Y,
	Non_Linear_Regression::control& ctrl, LossFunctions::error& res)
{
	valarray<double> tmp= (valarray<double>) f(X, beta) - Y;

	valarray<double> tmp2(tmp.size());
	res.main_error = 0;
	res.errorarray.resize(tmp.size());
	switch (ctrl.lossfunction)
	{
	case LossFunctions::huberlossfunction:
	{
		for (size_t i = 0; i < tmp.size(); i++)
		{
			double a = Statisticfunctions::fabs(tmp[i]);
			tmp2[i] = (a <= ctrl.huberslossfunction_border) ? 0.5 * a * a / tmp.size() : ctrl.huberslossfunction_border * (a - 0.5 * ctrl.huberslossfunction_border) / tmp.size();
			res.main_error += tmp2[i];
			res.errorarray[i] = tmp2[i];
		}
		break;
	}
	case LossFunctions::squaredresidual:
	{
		for (size_t i = 0; i < tmp.size(); i++)
		{
			tmp2[i] = tmp[i] * tmp[i] / tmp.size();
			res.main_error += tmp2[i];
			res.errorarray[i] = tmp2[i];
		}
		break;
	}
	case LossFunctions::absolutevalue:
	{
		for (size_t i = 0; i < tmp.size(); i++)
		{
			tmp2[i] = Statisticfunctions::fabs(tmp[i]) / tmp.size();
			res.main_error += tmp2[i];
		    res.errorarray[i] = tmp2[i];
		}
		break;
	}

	}
	return res.main_error;

	}
