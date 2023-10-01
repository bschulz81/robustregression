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


#include "statisticfunctions.h"
#include "matrixcode.h"
#include "linearregression.h"
#include "robustregression.h"
#include "nonlinearregression.h"
#include "lossfunctions.h"
#include <vector>

// An example of a linear function f(X,beta)=beta[0]*X+beta[1] to be fitted by the Levenberg-Marquardt algorithm
valarray<double> linear(const valarray<double>&X,const  valarray<double>& beta)
{
	valarray<double> Y(X.size());
	for (size_t i = 0; i < X.size(); i++)
		Y[i] = beta[0] * X[i] + beta[1];
	return Y;
}

// The Jacobi Matrix of the linear function f(X,beta)=beta[0]*X+beta[1]
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


//custom error function per point. Here the square of the residuals is chosen.
//the residuals are scaled by the pointnumber, in order to avoid that a smaller pointnumber always yields a smaller total error.
double err_pp(const double Y, double fY, const size_t pointnumber)
{
	return ((Y - fY)* (Y - fY)) /(double) pointnumber;
}
//the total error, given an array of residuals
double aggregate_err(valarray<double>& err)
{
	return err.sum();
}



int main(int argc, char* argv[])
{
	printf("We have 5!= ");
	printf("%f", Statisticfunctions::factorial(5));
	printf("\n");
	printf(" Gaussianalgorithm result example ");

	Matrix m(3, 3);
	m(0, 0) = 1;
	m(0, 1) = 2;
	m(0, 2) = -1;

	m(1, 0) = 1;
	m(1, 1) = 1;
	m(1, 2) = -1;
	m(2, 0) = 2;
	m(2, 1) = -1;
	m(2, 2) = 1;

	Vector v(3);
	v(0) = 2;
	v(1) = 0;
	v(2) = 3;

	Matrixcode::printvector(Matrixcode::Gaussian_algorithm(m, v));

	printf("\nDefine some array ");
	valarray<double>X0 = { 3, 13, 7, 5, 21, 23, 39, 23, 40, 23, 14, 12, 56, 23 };
	Matrixcode::printvector(Vector(X0));

	printf("\n compute the median ");
	printf("%f",Statisticfunctions::median(X0));

	printf("\n compute Q1 ");
	printf("%f", Statisticfunctions::Q1(X0));

	printf("\n compute Q3 ");
	printf("%f", Statisticfunctions::Q3(X0));


	valarray<double> X = { -3, 5,7, 10,13,16,20,22 };
	valarray<double> Y = { -210, 430, 590,830,1070,1310,1630,1790 };

	printf("\n\n\nSet the data on the X axis to: ");
	Matrixcode::printvector(Vector(X));
	printf("Set the data on the Y axis to: ");
	Matrixcode::printvector(Vector(Y));
	Linear_Regression::result res;
	Linear_Regression::linear_regression(X, Y, res);
	printf("\n\n\nSimple Linear regression\n");
	printf(" Slope ");
	printf("%f", res.main_slope);
	printf("\n Intercept  ");
	printf("%f", res.main_intercept);

	Linear_Regression::result res2;
	Linear_Regression::median_linear_regression(X, Y, res2);
	printf("\n\nMedian Linear regression \n");
	printf(" Slope ");
	printf("%f", res2.main_slope);
	printf("\n Intercept ");
	printf("%f", res2.main_intercept);

	valarray<double> X2 = { -3, 5,7, 10,13,15,16,20,22,25 };
	valarray<double> Y2 = { -210, 430, 590,830,1070,20,1310,1630,1790,-3 };
	printf("\n\nInsert 2 outliers at index 5 and 9 (X=15) set the data on the X axis to: \n");
	Matrixcode::printvector(Vector(X2));
	printf("\nSet the data wih the outliers at index 5 and 9 on the Y axis to: \n");
	Matrixcode::printvector(Vector(Y2));



	printf("\n\nSimple Linear regression with 2 inserted outliers \n");
	Linear_Regression::result res3;
	Linear_Regression::linear_regression(X2, Y2, res3);
	
	printf(" Slope ");
	printf("%f", res3.main_slope);
	printf("\n Intercept ");
	printf("%f", res3.main_intercept);


	printf("\n\nSiegel's repeated Median Linear regression with 2 inserted outliers \n");
	Linear_Regression::result res4;
	Linear_Regression::median_linear_regression(X2, Y2, res4);

	printf(" Slope ");
	printf("%f", res4.main_slope);
	printf("\n Intercept ");
	printf("%f", res4.main_intercept);

	printf("\n\n\nRobust regression with the same 2 inserted outliers\n");
	printf("\nModified last trimmed squares, with the default S estimator\n");

	Robust_Regression::modified_lts_control_linear ctrl5;
	Robust_Regression::linear_algorithm_result res5;
	Robust_Regression::modified_lts_regression_linear(X2, Y2, ctrl5, res5);
	
	printf(" Slope ");
	printf("%f", res5.main_slope);
	printf("\n Intercept ");
	printf("%f", res5.main_intercept);
	printf("\n Indices of outliers ");



	printf("\n\n\nRobust regression with the same 2 inserted outliers, but now with the Q Estimator and the loss function is quantile\n");

	Robust_Regression::modified_lts_control_linear ctr6;
	Robust_Regression::linear_algorithm_result res6;
	ctr6.rejection_method = Robust_Regression::tolerance_is_decision_in_Q_ESTIMATION;
	ctr6.lossfunction = LossFunctions::quantile;

	Robust_Regression::modified_lts_regression_linear(X2, Y2, ctr6, res6);

	
	printf("\nModified last trimmed squares\n");
	printf(" Slope ");
	printf("%f", res6.main_slope);
	printf("\n Intercept ");
	printf("%f", res6.main_intercept);
	printf("\n Indices of outliers ");

	for (size_t i = 0; i < res6.indices_of_removedpoints.size(); i++)
	{
		size_t w = res6.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}

		printf("\n\n modified least trimmed squares with the same 2 dataset but instead of the default S estimator, now the interquartile range method\n");

	Robust_Regression::modified_lts_control_linear ctrl7;
	Robust_Regression::linear_algorithm_result res7;


	ctrl7.rejection_method = Robust_Regression::tolerance_is_interquartile_range;
	ctrl7.outlier_tolerance = 1.5;
	Robust_Regression::modified_lts_regression_linear(X2, Y2, ctrl7, res7);

	printf(" Slope ");
	printf("%f", res7.main_slope);
	printf("\n Intercept ");
	printf("%f", res7.main_intercept);
	printf("\n Indices of outliers ");

	for (size_t i = 0; i < res7.indices_of_removedpoints.size(); i++)
	{
		size_t w = res7.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}


	printf("\n\nIterative outlier removal\n");


	Robust_Regression::linear_algorithm_result res8;
	Robust_Regression::linear_algorithm_control ctrl8;
	Robust_Regression::iterative_outlier_removal_regression_linear(X2, Y2, ctrl8, res8);
	printf(" Slope ");
	printf("%f", res8.main_slope);
	printf("\n intercept ");
	printf("%f", res8.main_intercept);

	printf("\n indices of outliers ");
	for (size_t i = 0; i < res8.indices_of_removedpoints.size(); i++)
	{
		size_t w = res8.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}


	Non_Linear_Regression::result res9;
	Non_Linear_Regression::control ctrl9;
	Non_Linear_Regression::initdata init9;
	init9.f = linear;
	init9.J = Jacobi;

	valarray<double>beta = { 1,1 };
	init9.initialguess = beta;


	printf("\n\n\nSimple Nonlinear Regression with the original dataset");

	Non_Linear_Regression::non_linear_regression(X, Y, init9, ctrl9, res9);

	printf("\n Slope ");
	printf("%f", res9.beta[0]);
	printf("\n intercept ");
	printf("%f", res9.beta[1]);


	printf("\n\n\nSimple Nonlinear Regression with the same 2 outliers added ");

	Non_Linear_Regression::result res10;
	Non_Linear_Regression::initdata init10;
	Non_Linear_Regression::control ctrl10;
	init10.f = linear;
	init10.J = Jacobi;
	init10.initialguess = beta;

	Non_Linear_Regression::non_linear_regression(X2, Y2, init10, ctrl10, res10);

	printf("\n Slope ");
	printf("%f", res10.beta[0]);
	printf("\n Intercept ");
	printf("%f", res10.beta[1]);




	printf("\n\n\nRobust non-linear regression with the same 2 inserted outliers\n");


	printf("\nModified last trimmed squares for non-linear regression\n");
	Robust_Regression::modified_lts_control_nonlinear ctrl11;
	Robust_Regression::nonlinear_algorithm_result res11;
	Non_Linear_Regression::initdata init11;
	init11.f = linear;
	init11.J = Jacobi;
	init11.initialguess = beta;
	Robust_Regression::modified_lts_regression_nonlinear(X2, Y2, init11, ctrl11, res11);

	
	
	printf(" Slope ");
	printf("%f", res11.beta[0]);
	printf("\n Intercept ");
	printf("%f", res11.beta[1]);
	printf("\n Indices of outliers ");

	for (size_t i = 0; i < res11.indices_of_removedpoints.size(); i++)
	{
		size_t w = res11.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}
	


	printf("\nModified last trimmed squares for non-linear regression with Huber's loss function\n\n");

	Robust_Regression::modified_lts_control_nonlinear ctrl12;
	ctrl12.lossfunction = LossFunctions::huberlossfunction;
	Robust_Regression::nonlinear_algorithm_result res12;
	Non_Linear_Regression::initdata init12;
	init12.f = linear;
	init12.J = Jacobi;
	init12.initialguess = beta;
	Robust_Regression::modified_lts_regression_nonlinear(X2, Y2, init12, ctrl12, res12);





	printf(" Slope ");
	printf("%f", res12.beta[0]);
	printf("\n Intercept ");
	printf("%f", res12.beta[1]);
	printf("\n Indices of outliers ");

	for (size_t i = 0; i < res12.indices_of_removedpoints.size(); i++)
	{
		size_t w = res12.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}

	printf("\n\nModified last trimmed squares for non-linear regression with a custom error function. Note that with non-linear regression algorithms, the loss "
		"function is now used both for outlier detection and for the curve fit\n");

	Robust_Regression::modified_lts_control_nonlinear ctrl13;
	ctrl13.loss_perpoint = err_pp;// custom loss function per point

	//if the aggregate error would not be defined here, the results of the calls of the loss functions per point would just be summed.
	ctrl13.aggregate_err = aggregate_err; 
	

	Robust_Regression::nonlinear_algorithm_result res13;
	Non_Linear_Regression::initdata init13;
	init13.f = linear;
	init13.J = Jacobi;
	init13.initialguess = beta;
	Robust_Regression::modified_lts_regression_nonlinear(X2, Y2, init13, ctrl13, res13);


	printf(" Slope ");
	printf("%f", res13.beta[0]);
	printf("\n Intercept ");
	printf("%f", res13.beta[1]);
	printf("\n Indices of outliers ");

	for (size_t i = 0; i < res13.indices_of_removedpoints.size(); i++)
	{
		size_t w = res13.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}
	printf("\n\nIterative outlier removal\n");



	Robust_Regression::nonlinear_algorithm_result res14;
	Robust_Regression::nonlinear_algorithm_control ctrl14;
	Non_Linear_Regression::initdata init14;
	init14.f = linear;
	init14.J = Jacobi;
	init14.initialguess = beta;

	Robust_Regression::iterative_outlier_removal_regression_nonlinear(X2, Y2, init14, ctrl14, res14);
	printf(" Slope ");
	printf("%f", res14.beta[0]);
	printf("\n Intercept ");
	printf("%f", res14.beta[1]);
	printf("\n Indices of outliers ");

	for (size_t i = 0; i < res14.indices_of_removedpoints.size(); i++)
	{
		size_t w = res14.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}



	return 0;
}
