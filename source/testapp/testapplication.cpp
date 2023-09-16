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
#include "Matrixcode.h"
#include "linearregression.h"
#include "robustregression.h"
#include "nonlinearregression.h"

#include <vector>

valarray<double> linear(const valarray<double>&X,const  valarray<double>& beta)
{
	valarray<double> Y(X.size());
	for (size_t i = 0; i < X.size(); i++)
		Y[i] = beta[0] * X[i] + beta[1];
	return Y;
}
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


	Linear_Regression::result res3;
	Linear_Regression::linear_regression(X2, Y2, res3);
	printf("\n\nSimple Linear regression with 2 inserted outliers \n");
	printf(" Slope ");
	printf("%f", res3.main_slope);
	printf("\n Intercept ");
	printf("%f", res3.main_intercept);

	Linear_Regression::result res4;
	Linear_Regression::median_linear_regression(X2, Y2, res4);
	printf("\n\nMedian Linear regression with 2 inserted outliers \n");
	printf(" Slope ");
	printf("%f", res4.main_slope);
	printf("\n Intercept ");
	printf("%f", res4.main_intercept);

	Robust_Regression::modified_lts_control_linear ctrl;
	Robust_Regression::linear_algorithm_result res5;
	Robust_Regression::modified_lts_regression_linear(X2, Y2, ctrl, res5);
	printf("\n\n\nRobust regression with the same 2 inserted outliers\n");
	printf("\nModified last trimmed squares, with the default S estimator\n");
	printf(" Slope ");
	printf("%f", res5.main_slope);
	printf("\n Intercept ");
	printf("%f", res5.main_intercept);
	printf("\n Indices of outliers ");


	Robust_Regression::modified_lts_control_linear ctrla;
	Robust_Regression::linear_algorithm_result res5a;
	ctrla.rejection_method = Robust_Regression::tolerance_is_decision_in_Q_ESTIMATION;
	Robust_Regression::modified_lts_regression_linear(X2, Y2, ctrla, res5a);
	printf("\n\n\nRobust regression with the same 2 inserted outliers, but now with the Q Estimator\n");
	printf("\nModified last trimmed squares\n");
	printf(" Slope ");
	printf("%f", res5a.main_slope);
	printf("\n Intercept ");
	printf("%f", res5a.main_intercept);
	printf("\n Indices of outliers ");




	for (size_t i = 0; i < res5a.indices_of_removedpoints.size(); i++)
	{
		size_t w = res5a.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}

	Robust_Regression::modified_lts_control_linear ctrlb;
	Robust_Regression::linear_algorithm_result res5b;


	ctrlb.rejection_method = Robust_Regression::tolerance_is_interquartile_range;
	ctrlb.outlier_tolerance = 1.5;
	Robust_Regression::modified_lts_regression_linear(X2, Y2, ctrlb, res5b);
	printf("\n\nRobust regression with the same 2 dataset but instead of the default S estimator, now the interquartile range method\n");
	printf("\nModified last trimmed squares\n");
	printf(" Slope ");
	printf("%f", res5b.main_slope);
	printf("\n Intercept ");
	printf("%f", res5b.main_intercept);
	printf("\n Indices of outliers ");

	for (size_t i = 0; i < res5b.indices_of_removedpoints.size(); i++)
	{
		size_t w = res5b.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}




	printf("\n\nIterative outlier removal\n");


	Robust_Regression::linear_algorithm_result res6;
	Robust_Regression::iterative_outlier_removal_regression_linear(X2, Y2, ctrl, res6);
	printf(" Slope ");
	printf("%f", res6.main_slope);
	printf("\n intercept ");
	printf("%f", res6.main_intercept);

	printf("\n indices of outliers ");
	for (size_t i = 0; i < res6.indices_of_removedpoints.size(); i++)
	{
		size_t w = res6.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}


	Non_Linear_Regression::result res7;
	Non_Linear_Regression::control ctrl2;
	Non_Linear_Regression::initdata init;
	init.f = linear;
	init.J = Jacobi;

	valarray<double>beta = { 1,1 };
	init.initialguess = beta;


	printf("\n\n\nSimple Nonlinear Regression with the original dataset");

	Non_Linear_Regression::non_linear_regression(X, Y, init, ctrl2, res7);

	printf("\n Slope ");
	printf("%f", res7.beta[0]);
	printf("\n intercept ");
	printf("%f", res7.beta[1]);


	printf("\n\n\nSimple Nonlinear Regression with the same 2 outliers added ");

	Non_Linear_Regression::result res8;
	Non_Linear_Regression::non_linear_regression(X2, Y2, init, ctrl2, res8);

	printf("\n Slope ");
	printf("%f", res8.beta[0]);
	printf("\n Intercept ");
	printf("%f", res8.beta[1]);


	Robust_Regression::modified_lts_control_nonlinear ctrl3;
	Robust_Regression::nonlinear_algorithm_result res9;
	Non_Linear_Regression::initdata init9;

	
	Robust_Regression::modified_lts_regression_nonlinear(X2, Y2, init, ctrl3, res9);

	printf("\n\n\nRobust non-linear regression with the same 2 inserted outliers\n");



	printf("\nModified last trimmed squares\n");
	printf(" Slope ");
	printf("%f", res9.beta[0]);
	printf("\n Intercept ");
	printf("%f", res9.beta[1]);
	printf("\n Indices of outliers ");

	for (size_t i = 0; i < res9.indices_of_removedpoints.size(); i++)
	{
		size_t w = res9.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}
	printf("\n\nIterative outlier removal\n");

	
	Robust_Regression::nonlinear_algorithm_result res10;


	Robust_Regression::iterative_outlier_removal_regression_nonlinear(X2, Y2, init, ctrl3, res10);
	printf(" Slope ");
	printf("%f", res10.beta[0]);
	printf("\n Intercept ");
	printf("%f", res10.beta[1]);
	printf("\n Indices of outliers ");

	for (size_t i = 0; i < res10.indices_of_removedpoints.size(); i++)
	{
		size_t w = res10.indices_of_removedpoints[i];
		printf("%lu", (unsigned long)w);
		printf(", ");
	}



	return 0;
}
