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


#include <valarray>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include "statisticfunctions.h"
#include "linearregression.h"
#include "nonlinearregression.h"
#include "matrixcode.h"
#include "robustregression.h"
namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(pyRobustRegressionLib, m) {
	m.doc() = "\n\nThe module pyRobustRegressionLib contains some functions for statistical estimation and robust linear and non-linear regression.\n"
		"It contains several sub-modules.\n\n The sub-module StatisticFunctions contains functions that are used by the robust regression algorithms but may "
		"also be used generally.\n\n The sub-module LinearRegression contains two functions for simple linear and the more robust median linear regression.\n\n"
		"The sub-module MatrixCode contains a Matrix and a Vector class for simple computations with Matrices and Vectors, e.g. the Gaussian algorithm.\n\n "
		"The sub-module NonLinearRegression contains an implementation of the Levenberg-Marquardt algorithm, together with classes to control it's behavior and "
		"classes for its initial and output data.\n\n The sub-module RobustRegression contains two different robust regression algorithms. Each of the algorithms "
		"are implemented with one version vor linear and another version for non - linear regression.\n\n Classes that are used to control the behavior of the " 
		"algorithms, as well as classes for the output data and the init data that must be specified for the non - linear regression algorithm are also provided";
	
	py::module Statisticfunctions = m.def_submodule("StatisticFunctions");

	Statisticfunctions.def("factorial", &Statisticfunctions::factorial,
		"Computes the factorial n!");

	Statisticfunctions.def("stdeviation", &Statisticfunctions::stdeviation,
		"Computes the standard deviation of an array");

	Statisticfunctions.def("average", &Statisticfunctions::average,
		"Computes the average of an array");

	Statisticfunctions.def("lowmedian", &Statisticfunctions::lowmedian,
		"Computes the low median of an array. In case of an even sized array, the item just below the middle, that one would get if the array were sorted, "
		"is returned.If the array size is odd, the middle of the array that one would get for a sorted array is returned.");

	Statisticfunctions.def("t", &Statisticfunctions::t,
		"Computes the Student t distribution for a significance level alpha and an array of size nu from the "
		"algorithm in Smiley W. Cheng, James C. Fu, Statistics & Probability Letters 1 (1983), 223-227");
	Statisticfunctions.def("crit", &Statisticfunctions::crit,
		"Computes the critical values of the student t distribution for"
		"significance level alpha and an array  of size N");

	Statisticfunctions.def("peirce", &Statisticfunctions::peirce,
		"Computes the peirce criterium from the point number, the number of outliers and the number of "
		"parameters to be fitted.\n see https://en.wikipedia.org/wiki/Peirce%27s_criterion for an " 
		"introduction and references to Peirce's original article in \n B. Peirce  Astronomical Journal II 45 (1852) ");

	Statisticfunctions.def("binomial", &Statisticfunctions::binomial,
		"Computes the binomial coefficient n over k");

	Statisticfunctions.def("Q_estimator", &Statisticfunctions::Q_estimator,
		"Computes the Q estimator of Croux and Rousseuuw for an array. \n"
		"The estimator was published in \n Peter J. Rousseeuw, Christophe Croux, Alternatives to the "
		"Median-Absolute Deviation\n J. of the Amer. Statistical Assoc. (Theory and Methods), "
		"88 (1993),p. 1273, and\n Croux, C., Rousseeuw, P.J. (1992). Time-Efficient Algorithms for "
		"Two Highly Robust Estimators of Scale.\n In: Dodge, Y., Whittaker, J. (eds) Computational Statistics. "
		"Physica, Heidelberg.\n https://doi.org/10.1007/978-3-662-26811-7_58 ");

	Statisticfunctions.def("S_estimator", &Statisticfunctions::S_estimator,
		"Computes the S estimator of Croux and Rousseuuw for an array. \n The estimator was published "
		"in \n Peter J.Rousseeuw, Christophe Croux, Alternatives to the Median - Absolute Deviation\n J.of "
		"the Amer.Statistical Assoc. (Theory and Methods), 88 (1993), p. 1273, and \n Croux, C., Rousseeuw, "
		"P.J. (1992).Time - Efficient Algorithms for Two Highly Robust Estimators of Scale.\n"
		"In : Dodge, Y., Whittaker, J. (eds)Computational Statistics.Physica, Heidelberg.\n"
		"https ://doi.org/10.1007/978-3-662-26811-7_58 (For the faster version of the S-estimator.)\n"
		"The version of the S and Q estimators in this library are now  adapted from Croux and Rousseeuw to "
		"the C language. \n Note that it is not the same Code because of some optimizations. Since many "
		"variables act on array indices in this algorithm, \n it was actually non - trivial to convert from "
		"Fortran to C.");

	Statisticfunctions.def("MAD_estimator", &Statisticfunctions::MAD_estimator,
		"Computes the MAD estimator for an array. The correction coefficients were from "
		"Croux, C., Rousseeuw, P.J. (1992), Time-Efficient Algorithms for Two Highly Robust Estimators of "
		"Scale.\n In: Dodge, Y., Whittaker, J. (eds) Computational Statistics. Physica, "
		"Heidelberg.\n https://doi.org/10.1007/978-3-662-26811-7_58 ");

	Statisticfunctions.def("T_estimator", &Statisticfunctions::T_estimator,
		"Computes the Q estimator of Croux and Rousseuuw for an array. \nThe estimator was published " 
		"in \n Peter J. Rousseeuw, Christophe Croux, Alternatives to the Median-Absolute Deviation\n J. of the "
		"Amer. Statistical Assoc. (Theory and Methods), 88 (1993),p. 1273, and\n Croux, C., Rousseeuw, P.J. "
		"(1992). Time-Efficient Algorithms for Two Highly Robust Estimators of Scale.\n In: Dodge, Y., "
		"Whittaker, J. (eds)Computational Statistics.Physica, Heidelberg.\n"
		"https ://doi.org/10.1007/978-3-662-26811-7_58 ");

	Statisticfunctions.def("onestepbiweightmidvariance", &Statisticfunctions::onestepbiweightmidvariance,
		"Computes the biweight midvariance for one step for an array.It expects the median m of the array. "
		"The estimator is described in \n T.C.Beers, K.Flynn and K.Gebhardt, Astron.J. 100 (1), 32 (1990)");

	Statisticfunctions.def("fabs", &Statisticfunctions::fabs,
		"Computes the absolute value of f");

	Statisticfunctions.def("median", &Statisticfunctions::median,
		"Computes the median of an array");

	Statisticfunctions.def("Q1", &Statisticfunctions::Q1,
		"Compute the Quartile Q1");

	Statisticfunctions.def("Q3", &Statisticfunctions::Q3,
		"Compute the Quartile Q3");

	Statisticfunctions.doc() = "Contains a set of statistic functtions\n factorial:Computes the factorial n!\n\n stdeviation : Computes the standard deviation of an "
				"array\n\n average : Computes the average of an array\n\n lowmedian : lowmedian\n\n t : "
				"Computes the Student t distribution for a significance level alpha and an array of size nu from the algorithm "
				"in Smiley W.Cheng, James C.Fu, Statistics& Probability Letters 1 (1983), 223 - 227\n\n crit : Computes the critical values "
				"of the student t distribution for significance level alpha and an array of size N\n\n peirce : Computes the peirce criterium "
				"from the point number, the number of outliers and the number of parameters to be fitted.\n see https "
				"://en.wikipedia.org/wiki/Peirce%27s_criterion for an introduction and references to Peirce's original article in \n B. "
				"Peirce  Astronomical Journal II 45 (1852) \n\n binomial:Computes the binomial coefficient n over k\n\n Q_estimator : Computes "
				"the Q estimator of Croux and Rousseuuw for an array. \nThe estimator was published in \n Peter J.Rousseeuw, Christophe Croux, "
				"Alternatives to the Median - Absolute Deviation\n J.of the Amer.Statistical Assoc. (Theory and Methods), 88 (1993), p. 1273, " 
				"and \n Croux, C., Rousseeuw, P.J. (1992). Time - Efficient Algorithms for Two Highly Robust Estimators of Scale.\n In : Dodge, "
				"Y., Whittaker, J. (eds)Computational Statistics.Physica, Heidelberg.\n https ://doi.org/10.1007/978-3-662-26811-7_58 \n\n"
				"S_estimator: Computes the S estimator of Croux and Rousseuuw for an array. \nThe estimator was published in \n Peter J.Rousseeuw, "
				"Christophe Croux, Alternatives to the Median - Absolute Deviation\n J.of the Amer.Statistical Assoc. (Theory and Methods), "
				"88 (1993), p. 1273, and \n Croux, C., Rousseeuw, P.J. (1992).Time - Efficient Algorithms for Two Highly "
				"Robust Estimators of Scale.\n In : Dodge, Y., Whittaker, J. (eds)Computational Statistics.Physica, Heidelberg.\n"
				"https ://doi.org/10.1007/978-3-662-26811-7_58 (For the faster version of the S-estimator.)\n The version of the S and Q estimator "
				"in this library  are now adapted from Croux and Rousseeuw to the C language. \n Note that it is not the same Code because of some "
				"optimizations. Since many variables act on array indices in this algorithm, \n it was actually non-trivial to convert from Fortran "
				"to C.\n\n MAD_estimator:Computes the MAD estimator for an array.The correction coefficients were from Croux, "
				"C., Rousseeuw, P.J. (1992), Time - Efficient Algorithms for Two Highly Robust Estimators 	of Scale.\n In : Dodge, Y., "
				"Whittaker, J. (eds)Computational Statistics.Physica, Heidelberg.\n https ://doi.org/10.1007/978-3-662-26811-7_58 \n\n T_estimator: "
				"Computes the Q estimator of Croux and Rousseuuw for an array. \nThe estimator was published in \n Peter J.Rousseeuw, "
				"Christophe Croux, Alternatives to the Median - Absolute Deviation\n J.of the Amer.Statistical Assoc. (Theory and Methods), 88 "
				"(1993), p. 1273, and \n Croux, C., Rousseeuw, P.J. (1992).Time - Efficient Algorithms for Two Highly Robust Estimators of "
				" Scale.\n In : Dodge, Y., Whittaker, J. (eds)Computational Statistics.Physica, Heidelberg.\n "
				"https ://doi.org/10.1007/978-3-662-26811-7_58 \n\n onestepbiweightmidvariance: Computes the biweight midvariance for one step for "
				"an array.It expects the median m of the array.The estimator is described in \n T.C.Beers, K.Flynn and K.Gebhardt, "
				"Astron.J. 100 (1), 32 (1990)\n\n fabs : Computes the absolute value of f\n\n median : Computes the median of an "
				"array\n\n Q1 : Compute the Quartile Q1\n\n Q3 : Compute the Quartile Q3";

	py::module linearregression = m.def_submodule("LinearRegression");

	linearregression.doc() = "Contains two functions, linear_regression and the more robust but slower median_linear_regression for linear curve fits. "
				"Both store their results for the main_slope and main_intercept in a struct called result";

	py::class_<Linear_Regression::result>(linearregression, "result")
		.def(py::init<>())

		.def_readwrite("main_slope", &Linear_Regression::result::main_slope,
			"Stores the resulting slope of the linear fit")

		.def_readwrite("main_intercept", &Linear_Regression::result::main_intercept,
			"Stores the resulting intercept of the linear fit ")

		.doc()="Stores the result of the linear regression in the variables main_slope and main_intercept"
		;
	
	linearregression.def("median_linear_regression", &Linear_Regression::median_linear_regression,
		"computes a linear regression. datapoints are two valarrays x and y.the result is put into res.");

	linearregression.def("linear_regression", &Linear_Regression::linear_regression,
		"Computes a median linear regression, which is more robust against outliers. "
		"Parameters are similar as in the linear regression. \nIt is slower than linear "
		"regression and for large and many outliers, it also does not yield precise results. ");

	py::module matrixcode = m.def_submodule("MatrixCode");

	matrixcode.doc() = "Contains two classes, Matrix and Vector. They are needed for the non-linear curve fitting algorithms and relatively basic, "
			"but they can be converted easily to other python arrays.It also contains the functions; Identity, which yields an identity "
			"matrix\n Transpose, which Yields the transpose of a matrix\nDiagonal, which yields the diagonal of a matrix, with all other "
			"entries put to zero.\n	Gaussian_algorithm that expects a Matrix and a Vector or an array and returns a Vector or an Array that "
			"is the result of the Gaussian algorithm.";

	py::class_<Vector>(matrixcode, "Vector")
		.def(py::init< valarray<double>>(), "Initializes a vector with elements set to the values of an array")

		.def(py::init< const  size_t >(),"Initializes a Vector with a given size and elements set to zero")

		.def("size", &Vector::Size,"Returns the size of a Vector")

		.def("resize", &Vector::Resize,"Resizes a Vector and sets its elements to zero")

		.def("__setitem__", [](Vector& self, size_t index, double val)

			{ self(index) = val; },"Sets an element in a Vector by an index beginning with 0, example: Vector[element]=X")
		.def("__getitem__", [](Vector& self, size_t index)
			{ return self(index); },"Gets the Element of a Vector by an index beginning with 0, example X=Vector[element]")

		.def(py::self + py::self,"Adds two vectors")

		.def(py::self - py::self,"Subtracts two Vectors")

		.def(py::self * double(),"Multiplies a scalar with a Vector")

		.def(py::self * py::self,"Multiplies two vectors (dot product)")

		.def(py::self / double(),"Dividesa vector by a scalar")

		.def("Printvector", [](Vector& self) {
		std::ostringstream os;
		for (size_t j = 0; j < self.Size(); j++)
		{
			os << self(j);
			os << " ";
		}
		std::string str(os.str());
		py::print(str);

		py::print("\n");
			}, "Prints a Vector")

		.doc() = "A Vector class\n The vector can be initialized by Vector(size), which initializes a vector with a given size of elements set to zero, "
				"or with Vector(array) which converts the array into an obhect of the vector class. \n size : Returns the size of a Vector\n resize : "
				"Resizes a Vector and sets its elements to zero\n items of the Vector can be set and get by[index], where index is zero based\n the "
				"Vector class supports the usual operations + , -.* for two Vectors and *and / for a Vector and a scalar\n Printvector : Prints a vector";
		;


	py::class_<Matrix>(matrixcode, "Matrix")

		.def(py::init<const size_t, const  size_t, valarray<double>>(),"Constructs a Matrix with a given number of rows and columns, expects an array, "
										"MatrixCode.Matrix(3,3,[1,2,-1,1,1,-1,2,-1,1]) yields a Matrix with elements "
										"m3[0, 0] = 1, m3[0, 1] = 2,m3[0, 2] = -1,m3[1, 0] = 1,m3[1, 1] = 1,m3[1, 2] = -1, "
										"m3[2, 0] = 2,m3[2, 1] = -1,m3[2, 2] = 1")

		.def(py::init<const size_t, const size_t>(), "constructs a Matrix with a given number of rows and columns, with elements set to 0")

		.def("Rows", &Matrix::Rows,"returns the number of rows of the Matrix")

		.def("Columns", &Matrix::Columns, "returns the number of columns of the Matrix")

		.def("SwapRows", &Matrix::SwapRows,"interchanges two rows of the Matrix with a given index beginning at 0.")

		.def("resize", &Matrix::Resize,"resizes the Matrix and sets it elements to zero")

		.def("__setitem__", [](Matrix& self, vector<size_t> v, double val)
					{ self(v[0], v[1]) = val; },"set an element by its index, beginning with 0 ")

		.def("__getitem__", [](Matrix& self, vector<size_t> v)
					{ return self(v[0], v[1]);
			}, "set an element by its index, beginning with 0 ")

		.def(py::self + py::self, "adds two Matrices")

				.def(py::self - py::self, "subtracts two Matrices")

				.def(py::self* double(), "multiplies a scalar with a Matrix")

				.def(py::self* py::self, "Multiplies two Matrices")

				.def("__mul__", [](const Matrix& a, Vector& b) {return a * b; }, py::is_operator())

				.def("__mul__", [](const Matrix& a, valarray<double>& b) {return a * b;	}, py::is_operator())
				.def("Printmatrix", [](Matrix& self)
					{
						for (long i = 0; i < self.Rows(); i++)
						{
							std::ostringstream os;
							for (size_t j = 0; j < self.Columns(); j++)
							{
								os << self(i, j);
								os << " ";
							}
							std::string str(os.str());
							py::print(str);
						}
						py::print("\n ");
					}, "Prints a Matrix")

				.doc() = "A Matrix Class\n Matrix(rows,columns) constructs a Matrix with a given number of rows and columns and elements set to zero \n"
						"Matrix(rows, columns, array) constructs a Matrix with a given number of rows and columsn and elements filled with a one "
						"dimensional array.For example Matrix(3, 3, [1, 2, -1, 1, 1, -1, 2, -1, 1]) yields a\nMatrix with elements m3[0, 0] = 1, "
						"m3[0, 1] = 2, m3[0, 2] = -1, m3[1, 0] = 1, m3[1, 1] = 1, m3[1, 2] = -1, m3[2, 0] = 2, m3[2, 1] = -1, m3[2, 2] = 1\n Rows "
						": returns the number of rows of the Matrix\n Columns : returns the number of columns of the Matrix\nSwapRows : "
						"interchanges two rows of the Matrix with a given index\n resize : resizes the Matrix and sets it elements to zero\n"
						"Elements of the Matrix can be set and get by X = m[row, column] and m[row, column] = X\n The Matrix class supports the "
						"usual operations + , -*for two Matrices and *for a Matrix and a Vector and *for a Matrix and a scalar.\n Printmatrix : Prints a Matrix";
				;
		matrixcode.def("Identity", &Matrixcode::Identity,"Yields an identity matrix with a specified number of rows and columns");

		matrixcode.def("Transpose", &Matrixcode::Transpose,"Yields the transpose of a matrix");

		matrixcode.def("Diagonal", &Matrixcode::Diagonal,"Yields the diagonal of a matrix, with all other entries put to zero.");

		matrixcode.def("Gaussian_algorithm", static_cast<Vector(*)(const Matrix&, const Vector&)>(&Matrixcode::Gaussian_algorithm),
			"Solves a linear equation. Expects a matrix and a vector and returns a vector.");

		matrixcode.def("Gaussian_algorithm", static_cast<valarray<double>(*)(const Matrix&, const valarray<double>&)>(&Matrixcode::Gaussian_algorithm),
			"Solves a linear equation. expects a matrix and an array and returns an array.");

	py::module nonlinearregression = m.def_submodule("NonLinearRegression");

	nonlinearregression.doc() = "contains classes that control the non-linear regression algorithms. The class initdata is for the initialisation data of "
								"the algirithm, error contains the residuals, result stores the result,errorfunction specifies the loss function "
								"properties, control determines the algorithm behavior. The functions non_linear_regression make a non-linear curve "
								"fit with a Levenberg-Marquardt algorithm according to  https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm, the "
								"function  nonlinear_loss_function computes the residuals by some specified metric";

	
	py::class_< Non_Linear_Regression::initdata>(nonlinearregression, "initdata")
		.def(py::init<>())
		
		.def_readwrite("initialguess", &Non_Linear_Regression::initdata::initialguess, "Contains an array that represents the initial data for the non-linear fit")

		.def_readwrite("Jacobian", &Non_Linear_Regression::initdata::J, "Contains the Jacobi Matrix of the function f(X,beta) whose parameters beta are to be "
										"found and fitted to data X and Y.For example, for a linear function, the Jacobi Matrix is "
										"given by : def Jacobi(X, beta) : \n Matrix(len(X), len(beta)) \n	for i in "
										"range(0, len(X)) :\n m[i, 0] = X[i]\n	m[i, 1] = 1\n return m")

		.def_readwrite("f", &Non_Linear_Regression::initdata::f, "Contains the function f(X,beta) whose parameters beta are to be found and fitted to data X and Y. "
									"For example, for a linear function, f is given by : def linear(X, beta) : \n Y = []\n for i in "
									"range(0, len(X)) :\n	Y.append(beta[0] * X[i] + beta[1])\n return Y ")

		.doc()="Contains the initialisation data for the non-linear regression algorithms. An array called initialguess, a pointer to a Matrix function called "
		"Jacobian(X, beta), and a pointer to the function f(X, beta), where X is an array on the X axis and beta is an array for the parameters of f to be found in "
		"the curve fit"
		;
	
	py::class_< LossFunctions::error>(nonlinearregression, "error")
		.def(py::init<>())

		.def_readwrite("errorarray", &LossFunctions::error::errorarray, "Contains an array of the residuals, computed with respect to some metric, e.g. "
										"the squared differences, the absolute value of the differences, or Huber's loss function, "
										"between each data point, and the fitted function")

		.def_readwrite("main_error", &LossFunctions::error::main_error, "Contains a measure for the entire error of the fit with respect to the data, with respect "
										"to some metric, e.g the absolute values, the squared residuals or Huber's loss function")

		.doc()="Contains the residuals of the curve fit, as calculated by some metric. The residuals for each point are in errorarray, "
				"the entire residual is in main_error"
		;
	
	py::class_< Non_Linear_Regression::result>(nonlinearregression, "result")
		.def(py::init<>())

		.def_readwrite("beta", &Non_Linear_Regression::result::beta,
			"Stores the array of the parameters that are the result of the non-linear fit")

		.doc()="Contains the result of the non-linear regression in an array beta"
		;

	py::enum_< LossFunctions::errorfunction_name>(nonlinearregression, "errorfunction_name")
		.value("huberlossfunction", LossFunctions::errorfunction_name::huberlossfunction,
			"The loss function is given by Huber's loss function")

		.value("squaredresidual", LossFunctions::errorfunction_name::squaredresidual,
			"The loss function is given by the squared residuals")

		.value("absolutevalue", LossFunctions::errorfunction_name::absolutevalue,
			"The Loss function is given by the absolute values of the residuals")
		.export_values()
		;

	py::class_< LossFunctions::errorfunction>(nonlinearregression, "errorfunction")
		.def(py::init<>())
		.def_readwrite("lossfunction", &LossFunctions::errorfunction::lossfunction, 
			"The metric with which the loss function is calculated")

		.def_readwrite("huberslossfunction_border", &LossFunctions::errorfunction::huberslossfunction_border,
			"If Huber's loss function is used, this determines its borders delta")

		.doc()="Describes the metric how errors are computed. the field lossfunction describes the name of the metric for the errors. It can be  huberlossfunction, "
			"squaredresidual or absolutevalue. If Huber's loss function is chosen, the field huberslossfunction_border describes the delta parameter "
		      "of the loss function "
		;

	py::class_< Non_Linear_Regression::control, LossFunctions::errorfunction>(nonlinearregression, "control")
		.def(py::init<>())
		
		.def_readwrite("lambda", &Non_Linear_Regression::control::lambda,
			"The Lambda parameter for the Levenberg-Marquardt algorithm, see "
			"https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm . Usually the provideddefault works")

		.def_readwrite("increment", &Non_Linear_Regression::control::increment,
			"The increment parameter for the Levenberg-Marquardt algorithm,  see "
			"https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm . Usually the provided default works")

		.def_readwrite("decrement", &Non_Linear_Regression::control::decrement,
			"The decrement parameter for the Levenberg-Marquardt algorithm, "
			"see  https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm . Usually the provided default works")

		.def_readwrite("precision", &Non_Linear_Regression::control::precision,
			"Sets the precision of the result after which the iteration of the non-linear fit is stopped")

		.def_readwrite("h", &Non_Linear_Regression::control::h,
			"A parameter for some kind of directional derivative in the Levenberg-Marquardt algorithm, see "
			"https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm . Usually the provided default works")

		.def_readwrite("stop_nonlinear_curve_fitting_after_iterations", &Non_Linear_Regression::control::stop_nonlinear_curve_fitting_after_iterations, 
			"A parameter that stops the Levenberg-Marquardt algorithm after a given number of iterations")

		.def_readwrite("stop_nonlinear_curve_fitting_after_seconds", &Non_Linear_Regression::control::stop_nonlinear_curve_fitting_after_seconds,
			"A parameter that stops the Levenberg-Marquardt algorithm after a given number of seconds")

		.def_readwrite("tolerable_error", &Non_Linear_Regression::control::tolerable_error, 
			"Sets a tolerable error after which the iteration of the Levenberg-Marquardt algorithm stops")

		.doc()="Describes the parameters of the Levenberg-Marquardt algorithm with fields lambda, increlemt, decrement,h which are similar as in "
			"https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm additionally, it has the fields precision, which stop the algorithm after "
			"its improval is only increased in an iteration by a margin smaller than precision,  stop_nonlinear_curve_fitting_after_iterations "
			"which stop the algorithm after a given number of iterations, stop_nonlinear_curve_fitting_after_seconds, which stops the algorithm "
			"after a given time and	tolerable_error, which stops the algorithm once the error is deemed small enough by the given margin"
		;

	nonlinearregression.def("non_linear_regression", &Non_Linear_Regression::non_linear_regression,
		"Computes the non linear regression from given arrays X,Y, initdata with a function to be fit and a jacobian, and given "
		"controldata and yields the result");

	nonlinearregression.def("nonlinear_loss_function", &Non_Linear_Regression::nonlinear_loss_function,
		"Computes the loss function given the  loss function given the function f(X, beta), the arrays for X and beta, the data Y "
		"and the error metric.The output is stored in res");


	py::module robustregression = m.def_submodule("RobustRegression");
	robustregression.doc() = "contains several classes needed for the initialisation and control of various robust regression algorithms\n The robust regression "
				"algorithms in the module are divided into linear and non-linear robust regression algorithms\n	modified_lts_regression_linear "
				"is a modified forward search algorithm.A subset of k points of the n points is chosen.A linear regression is made with the k "
				"points.Then, it is computed whether\n	the other n - k points are outliers by a specified robust estimator.If not, "
				"they are added to the fit and a new linear regression is made.Then a different set of k points is chosen and the procedure "
				"is repeated until the best model is found.\n	The function iterative_outlier_removal_regression_linear uses a specified robust "
				"estimator to find whether a point is an outlier based on its residual.Then, the point is removed and another\n	linear curve fit "
				"is made.This is done, until there are no outliers anymore in the data.\nThe functions modified_lts_regression_nonlinear and "
				"iterative_outlier_removal_regression_nonlinear implement the same robust algorithms for non - linear curve fits.\n	The function "
				"linear_loss_function computes the loss function for the curve fit and the data based on some metric, which can be squared residuals, "
				"absulute values of the residuals or Huber's loss function. \n	In order to control the algorithm, the module has various "
				"classes.\nmodified_lts_control_nonlinear and modified_lts_control_linear are used to control the non - linear and linear algorithms "
				"based on the forward search.\n	nonlinear_algorithm_control and linear_algorithm_control control the behaviors of the non - linear and "
				"linear iterative outlier removal algorithms.\n	nonlinear_algorithm_result and linear_algorithm_result are used to store the result of "
				"the non - linear and linear curve fitting algorithms\n	The non - linear regression algorithms also need the initdata class provided "
				"by the NonLinearRegression module\nThe classes contain several subclasses.  \n	For example, linear_algorithm_result and "
				"nonlinear_algorithm_result inherit the result class from the NonLinearRegression and LinearRegression	module, but also a class "
				"RobustRegression, which contains the indices of the poihts that are rejected and used by the curve fits the classes "
				"linear_algorithm_control inherits the class control which determines the robust estimators that are used for outlier detection. "
				"additionally  linear_algorithm_control inherits the class errorfunction, which determines the loss function used in the curve fit. "
				"\n	the classes nonlinear_algorithm_control inherits the class control which determines the robust estimators that are used for outlier "
				"detection.\nAdditionally  nonlinear_algorithm_control inherits the class control from the non - linear - regression module, "
				"which determines how the non - linear algorithm behaves\n the clases modified_lts_control_linear inherits linear_algorithm_control "
				"and modified_lts_control_nonlinear inherits the class nonlinear_algorithm_control but both classes also inherit the class\n "
				"lts_control which determines whether a ransac should be used or how much workload should be distributed to a thread.";

	py::class_< Robust_Regression::result, LossFunctions::error>(robustregression, "result")
		.def(py::init<>())

		.def_readwrite("indices_of_removedpoints", &Robust_Regression::result::indices_of_removedpoints,
			"Contains an array with the indices of the points that were removed by the robust fit")

		.def_readwrite("indices_of_used_points", &Robust_Regression::result::indices_of_used_points, 
			"Contains an array with the indices of the points that were used by the robust fit")

		.doc()="Describes the result from the robust procedures in terms of errorfunctions and removal of points.\n Inherits from LossFunctions::error \n The field "
			"indices_of_removedpoints Stores the indices of the removed points and indices_of_used_points stores the points that were used in the robust curve fits"
		;

	py::class_< Robust_Regression::linear_algorithm_result, Robust_Regression::result,Linear_Regression::result>(robustregression,
		"linear_algorithm_result")
		.def(py::init<>())

		.doc()="Stores the result of the linear curve fitting algorithms. Inherits from  RobustRegression::result and LinearRegression::result"
		;


	py::class_< Robust_Regression::nonlinear_algorithm_result,Robust_Regression::result,Non_Linear_Regression::result>(robustregression,
		"nonlinear_algorithm_result")
		.def(py::init<>())

		.doc() = "Stores the result of the non-linear curve fitting algorithms. Inherits from  RobustRegression::result and  LinearRegression::result "
		;

	py::enum_< Robust_Regression::estimator_name>(robustregression, "estimator_name")

		.value("no_rejection", Robust_Regression::estimator_name::no_rejection,"If set, no rejection happens")

		.value("tolerance_is_maximum_squared_error", Robust_Regression::estimator_name::tolerance_is_maximum_error," The tolerance parameter sets the maximally "
			"tolerable errorof a point, where error is computed by some metric, i.e.the absolute value of the residuals, the squared residuals or Huber's loss "
			"function. Choosing this estimator is not recommended since it is not robust.")

		.value("tolerance_multiplies_standard_deviation_of_error", Robust_Regression::estimator_name::tolerance_multiplies_standard_deviation_of_error,"a point with " 
			"an error is an outlier if | error - average(errors) | >tolerance * standard_deviation(errors).The errors are computed by some metric, i.e.the absolute "
			"value of the residuals, the squared residuals or Huber's loss function.It is not recommended to use this estimator since it is not robust")

		.value("tolerance_is_significance_in_Grubbs_test", Robust_Regression::estimator_name::tolerance_is_significance_in_Grubbs_test,"A point is an outlier if it "
			"its error is determined as such by the Grubbs test, given the distribution of the errors, with a significance defined the tolerance parameter.The "
			"errors are computed by some metric, i.e.the absolute value of the residuals, the squared residuals or Huber's loss function.")

		.value("tolerance_is_decision_in_MAD_ESTIMATION", Robust_Regression::estimator_name::tolerance_is_decision_in_MAD_ESTIMATION,"A point with an error is an "
			"outlier if |error-median(errors)|/MAD(errors)>tolerance. The errors are computed by some metric, i.e. the absolute value of the residuals, "
			"the squared residuals or Huber's loss function.  This estimator is robust and recommended")

		.value("tolerance_is_decision_in_S_ESTIMATION", Robust_Regression::estimator_name::tolerance_is_decision_in_S_ESTIMATION, "A point with an error is an "
			"outlier if | error - median(errors)|/S_estimator(errors)>tolerance.The errors are computed by some metric, i.e.the absolute value of the residuals, "
			"the squared residuals or Huber's loss function.  This estimator is robust and recommended")

		.value("tolerance_is_decision_in_Q_ESTIMATION", Robust_Regression::estimator_name::tolerance_is_decision_in_Q_ESTIMATION,"A point with an error is an outlier "
			"if | error - median(errors)|/Q_estimator(errors)>tolerance.The errors are computed by some metric, i.e.the absolute value of the residuals, the squared "
			"residuals or Huber's loss function.  This estimator is robust and recommended")

		.value("tolerance_is_decision_in_T_ESTIMATION", Robust_Regression::estimator_name::tolerance_is_decision_in_T_ESTIMATION,"A point with an error is an outlier "
			"if | error - median(errors)|/T_estimator(errors)>tolerance.The errors are computed by some metric, i.e.the absolute value of the residuals, the squared "
			"residuals or Huber's loss function.")

		.value("use_peirce_criterion", Robust_Regression::estimator_name::use_peirce_criterion,"If set, the Peirce criterion is used on the array of errors in order "
			"to establish whether a point is an outlier.The values of the errors are computed by some metric, i.e.the absolute value of the residuals, the squared "
			" residuals or Huber's loss function. " )

		.value("tolerance_is_biweight_midvariance", Robust_Regression::estimator_name::tolerance_is_biweight_midvariance,"A point with an error is an outlier "
			"if | error - median(errors)|/Biweight_midvariance(errors)>tolerance.The errors are computed by some metric, i.e.the absolute value of the residuals, "
			"the squared residuals or Huber's loss function.")

		.value("tolerance_is_interquartile_range", Robust_Regression::estimator_name::tolerance_is_interquartile_range, "A point is an outlier its error is smaller "
			"than Q1 - tolerance * Inter_Quartile_Range or larger than Q2 + tolerance * Inter_Quartile_Range. The values of the errors are computed "
			"by some metric, i.e. the absolute value of the residuals, the squared residuals or Huber's loss function.")
		.export_values()
		;

	py::class_< Robust_Regression::control>(robustregression, "control")
		.def(py::init<>())

		.def_readwrite("outlier_tolerance", &Robust_Regression::control::outlier_tolerance,
			"Sets the value of the tolerance parameter used by the robust regression")

		.def_readwrite("rejection_method", &Robust_Regression::control::rejection_method,
			"Sets the estimator for the outlier rejection")

		.def_readwrite("stop_after_seconds", &Robust_Regression::control::stop_after_seconds,
			"Stops the robust regression after a given number of seconds")

		.def_readwrite("stop_after_numberofiterations_without_improvement", &Robust_Regression::control::stop_after_numberofiterations_without_improvement,
			"Stops the robust regression after a given number of iterations with no improvement")

		.def_readwrite("maximum_number_of_outliers", &Robust_Regression::control::maximum_number_of_outliers,
			"Sets a maximum number of outliers that the algorithm can find.Per default or if set to zero, "
			"it will designate at maximum 30 % of the points as outliers ")

		.doc()="Controls the behavior of the robust regression algorithms,outlier_tolerance: Sets the value of the tolerance parameter used by the robust regression\n"
		"rejection_method: Sets the estimator for the outlier rejection stop_after_seconds : Stops the robust regression after a given number of seconds\n"
		"stop_after_numberofiterations_without_improvement : Stops the robust regression after a given number of iterations with no improvement  \n"
		"maximum_number_of_outliers : Sets a maximum number of outliers that the algorithm can find. Per default or if set to zero, it will designate at maximum 30 % "
		"of the points as outliers";
		;

	py::class_< Robust_Regression::linear_algorithm_control,Robust_Regression::control, LossFunctions::errorfunction>(robustregression, "linear_algorithm_control")
		.def(py::init<>())

		.def_readwrite("use_median_regression", &Robust_Regression::linear_algorithm_control::use_median_regression,
			"Sets whether the median linear regression algorithm should be used as a basis of the robust curve fitting algorithms")

		.def_readwrite("tolerable_error", &Robust_Regression::linear_algorithm_control::tolerable_error,
			"Quits the robust curve fitting if the error is less than the tolerable error set with this variable")

		.doc()="Controls the behavior of the robust linear regression algorithms\n inherits from  Robust_Regression::control and  LossFunctions::errorfunction. \n"
		"use_median_regression determines whether median regression should be used\n tolerable_error gives a margin for the residual where the algorithm stops if the"
		"residual becomes smaller"
		;

	py::class_< Robust_Regression::nonlinear_algorithm_control,	Robust_Regression::control, Non_Linear_Regression::control>(robustregression,
		"nonlinear_algorithm_control")

		.def(py::init<>())

		.doc()="Controls the behavior of the robust non-linear regression algorithms. "
		"Inherits from  Robust_Regression::control and  Non_Linear_Regression::control "
		;

	py::class_ < Robust_Regression::lts_control>(robustregression, "lts_control")

		.def(py::init<>())

		.def_readwrite("use_ransac", &Robust_Regression::lts_control::use_ransac,
			"Sets whether a Ransac algorithm should be used as robust fitting algorithm")

		.def_readwrite("workload_distributed_to_several_threads", &Robust_Regression::lts_control::workload_distributed_to_several_threads,
			"Sets the number of iterations that are distributed to a single thread")

		.doc()="Controls the behavior of the forward search algorithms. use_ransac determines whether the ransac should be used\n"
		"workload_distributed_to_several_threads determines the workload distributed to a single thread "
		;


	py::class_< Robust_Regression::modified_lts_control_linear, Robust_Regression::linear_algorithm_control, Robust_Regression::lts_control>(robustregression,
		"modified_lts_control_linear")
		.def(py::init<>())
		.doc()="Controls the behavior of the linear forward search algorithms. Inherits from Robust_Regression::modified_lts_control_linear, "
		"Robust_Regression::linear_algorithm_control, Robust_Regression::lts_control"
		;
	py::class_< Robust_Regression::modified_lts_control_nonlinear, Robust_Regression::nonlinear_algorithm_control, Robust_Regression::lts_control>(robustregression,
		"modified_lts_control_nonlinear")
		.def(py::init<>())
		.doc()="Controls the behavior of the non-linear forward search algorithms. Inherits from  Robust_Regression::modified_lts_control_nonlinear, "
		"Robust_Regression::nonlinear_algorithm_control, Robust_Regression::lts_control"
		;

	robustregression.def("modified_lts_regression_linear", &Robust_Regression::modified_lts_regression_linear,
		"Implements a modified least trimmed squares algorithm.\n"
		"Ít starts with a minimal model of a given size.It consists of a combination of points from the data and has the size "
		"pointnumber-maximum_number_of_outliers\n Then one makes a curve fit and looks whether the other points in the dataset would be outliers based on their "
		"residuals and statistical estimators.\n If the points are not outliers, they are added to the model.\n A curve fit with the enlarged model is done, and then "
		"one starts the process from another combination of pointnumber-maximum_number_of_outliers points\n The best fit that was found is given as a result. This "
		"function is for linear regression.");

	robustregression.def("iterative_outlier_removal_regression_linear", &Robust_Regression::iterative_outlier_removal_regression_linear,
		"Implements an iterative outlier removal algorithm for linear regression.\n One starts with a curve fit of all points. Then one looks "
		"at the data and estimates whether the residuals computed for certain points are outliers.\n These points are then removed, and another "
		"fit is made.The process continues until some specified maximum number of outliers is removed or no \n outliers are found anymore.");

	robustregression.def("modified_lts_regression_nonlinear", &Robust_Regression::modified_lts_regression_nonlinear, py::call_guard<py::gil_scoped_release>(),
		"Implements a modified least trimmed squares algorithm for non-linear curve fits. \n one starts with a minimal model of a given size. It consists of "
		"a combination of points from the data and has the size pointnumber-maximum_number_of_outliers \n Then one makes a curve fit and looks whether the other "
		"points in the dataset would be outliers. If not, they are added to the model. \n A curve fit with the enlarged model is done, and then one starts the "
		"process from another combination of pointnumber-maximum_number_of_outliers points \n The best fit that was found is given as a result. This function is for "
		"non - linear regression.");

	robustregression.def("iterative_outlier_removal_regression_nonlinear", &Robust_Regression::iterative_outlier_removal_regression_nonlinear,
		"Implements an iterative outlier removal algorithm for nonlinear regression.\n One starts with a curve fit of all points.Then one looks at the data and "
		"estimates whether the residuals computed for certain points are outliers.\n These points are then removed, and another fit is made.The process continues "
		"until some specified maximum number of outliers is removed or no\n outliers are found anymore.");

	robustregression.def("linear_loss_function", &Robust_Regression::linear_loss_function,
		"Implements a loss function for a linear curve fit that computes the residuals "
		"for every point given some metric (squared residuals, Huber's loss function, absolute value), and the sum of these residuals");
	};
