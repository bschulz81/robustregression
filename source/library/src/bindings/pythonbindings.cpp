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

	py::module Statisticfunctions = m.def_submodule("StatisticFunctions");
    float (*d1)(float) = &Statisticfunctions::fabs;
    double (*d2)(double) = &Statisticfunctions::fabs;
	Statisticfunctions.def("factorial", &Statisticfunctions::factorial);
	Statisticfunctions.def("stdeviation", &Statisticfunctions::stdeviation);
	Statisticfunctions.def("average", &Statisticfunctions::average);
	Statisticfunctions.def("lowmedian", &Statisticfunctions::lowmedian);
	Statisticfunctions.def("t", &Statisticfunctions::t);
	Statisticfunctions.def("crit", &Statisticfunctions::crit);
	Statisticfunctions.def("peirce", &Statisticfunctions::peirce);
	Statisticfunctions.def("binomial", &Statisticfunctions::binomial);
	Statisticfunctions.def("Q_estimator", &Statisticfunctions::Q_estimator);
	Statisticfunctions.def("S_estimator", &Statisticfunctions::S_estimator);
	Statisticfunctions.def("MAD_estimator", &Statisticfunctions::MAD_estimator);
	Statisticfunctions.def("T_estimator", &Statisticfunctions::T_estimator);
	Statisticfunctions.def("onestepbiweightmidvariance", &Statisticfunctions::onestepbiweightmidvariance);
	Statisticfunctions.def("fabs", d2);
	Statisticfunctions.def("fabs", d1);
    double (*d3)(valarray<double>arr, size_t n) = &Statisticfunctions::median;
    double (*d4)(valarray<float>arr, size_t n) = &Statisticfunctions::median;
	Statisticfunctions.def("median", d3);
	Statisticfunctions.def("median", d4);


	py::module linearregression = m.def_submodule("LinearRegression");

    py::class_<Linear_Regression::result>(linearregression, "result")
		.def(py::init<>())
        .def_readwrite("main_slope", &Linear_Regression::result::main_slope)
        .def_readwrite("main_intercept", &Linear_Regression::result::main_intercept);
        ;
	linearregression.def("median_linear_regression", &Linear_Regression::median_linear_regression);
	linearregression.def("linear_regression", &Linear_Regression::linear_regression);

	py::module matrixcode = m.def_submodule("MatrixCode");

	py::class_<Vector>(matrixcode, "Vector")
		.def(py::init< valarray<double>>())
		.def(py::init< const  size_t >())
		.def("size", &Vector::Size )
		.def("resize", &Vector::Resize)
		.def("__setitem__", [](Vector& self, size_t index, double val)
			{ self(index) = val; })
		.def("__getitem__", [](Vector& self, size_t index)
			{ return self(index); })
		.def(py::self + py::self)
		.def(py::self - py::self)
		.def(py::self * double())
		.def(py::self * py::self)
		.def(py::self / double())
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
			})
		;

	py::class_<Matrix>(matrixcode, "Matrix")
		.def(py::init<const size_t, const  size_t, valarray<double>>())
		.def(py::init<const size_t, const size_t>())
		.def("Rows", &Matrix::Rows)
		.def("Columns", &Matrix::Columns)
		.def("SwapRows", &Matrix::SwapRows)
		.def("resize", &Matrix::Resize)

		.def("__setitem__", [ ](Matrix& self, vector<size_t> v, double val)
			{ self(v[0], v[1]) = val; })
		.def("__getitem__", [](Matrix& self, vector<size_t> v)
			{ return self(v[0], v[1]);
})
		.def(py::self + py::self)
		.def(py::self - py::self)
		.def(py::self * double())
		.def(py::self * py::self)
		.def("__mul__", [](const Matrix& a, Vector& b) {
		return a * b;
			}, py::is_operator())

		.def("__mul__", [](const Matrix& a, valarray<double>& b) {
				return a * b;
			}, py::is_operator())

		.def("Printmatrix",[](Matrix& self) {
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
			})
		;

		matrixcode.def("Identity", &Matrixcode::Identity);
		matrixcode.def("Transpose", &Matrixcode::Transpose);
		matrixcode.def("Diagonal", &Matrixcode::Diagonal);
		matrixcode.def("Gaussian_algorithm", static_cast<Vector(*)(const Matrix&,const Vector&)>(&Matrixcode::Gaussian_algorithm));
		matrixcode.def("Gaussian_algorithm", static_cast<valarray<double>(*)(const Matrix &,const valarray<double>&)>(&Matrixcode::Gaussian_algorithm));



	py::module nonlinearregression = m.def_submodule("NonLinearRegression");

	py::class_< Non_Linear_Regression::initdata>(nonlinearregression, "initdata")
		.def(py::init<>())
		.def_readwrite("initialguess", &Non_Linear_Regression::initdata::initialguess)
		.def_readwrite("Jacobian", &Non_Linear_Regression::initdata::J)
		.def_readwrite("f", &Non_Linear_Regression::initdata::f)
		;
	
	py::class_< LossFunctions::error>(nonlinearregression, "error")
		.def(py::init<>())
		.def_readwrite("errorarray", &LossFunctions::error::errorarray)
		.def_readwrite("main_error", &LossFunctions::error::main_error)
		;

	py::class_< Non_Linear_Regression::result>(nonlinearregression, "result")
		.def(py::init<>())
		.def_readwrite("beta", &Non_Linear_Regression::result::beta)
		;

	py::enum_< LossFunctions::errorfunction_name>(nonlinearregression, "errorfunction_name")
		.value("huberlossfunction", LossFunctions::errorfunction_name::huberlossfunction)
		.value("squaredresidual", LossFunctions::errorfunction_name::squaredresidual)
		.value("absolutevalue", LossFunctions::errorfunction_name::absolutevalue)
		.export_values()
		;


	py::class_< LossFunctions::errorfunction>(nonlinearregression, "errorfunction")
		.def(py::init<>())
		.def_readwrite("lossfunction", &LossFunctions::errorfunction::lossfunction)
		.def_readwrite("huberslossfunction_border", &LossFunctions::errorfunction::huberslossfunction_border)
		;

	py::class_< Non_Linear_Regression::control, LossFunctions::errorfunction>(nonlinearregression, "control")
		.def(py::init<>())
		.def_readwrite("lambda", &Non_Linear_Regression::control::lambda)
		.def_readwrite("increment", &Non_Linear_Regression::control::increment)
		.def_readwrite("decrement", &Non_Linear_Regression::control::decrement)
		.def_readwrite("precision", &Non_Linear_Regression::control::precision)
		.def_readwrite("h", &Non_Linear_Regression::control::h)
		.def_readwrite("stop_nonlinear_curve_fitting_after_iterations", &Non_Linear_Regression::control::stop_nonlinear_curve_fitting_after_iterations)
		.def_readwrite("stop_nonlinear_curve_fitting_after_seconds", &Non_Linear_Regression::control::stop_nonlinear_curve_fitting_after_seconds)
		.def_readwrite("tolerable_error", &Non_Linear_Regression::control::tolerable_error)
		;

	nonlinearregression.def("non_linear_regression", &Non_Linear_Regression::non_linear_regression);
	nonlinearregression.def("nonlinear_loss_function", &Non_Linear_Regression::nonlinear_loss_function);

	py::module robustregression = m.def_submodule("RobustRegression");

	py::class_< Robust_Regression::result, LossFunctions::error>(robustregression, "result")
		.def(py::init<>())
		.def_readwrite("indices_of_removedpoints", &Robust_Regression::result::indices_of_removedpoints)
		.def_readwrite("indices_of_used_points", &Robust_Regression::result::indices_of_used_points)
		;

	py::class_< Robust_Regression::linear_algorithm_result, Robust_Regression::result,Linear_Regression::result>(robustregression, "linear_algorithm_result")
		.def(py::init<>())
		;
	py::class_< Robust_Regression::nonlinear_algorithm_result,Robust_Regression::result,Non_Linear_Regression::result>(robustregression, "nonlinear_algorithm_result")
		.def(py::init<>())
		;

	py::enum_< Robust_Regression::estimator_name>(robustregression, "estimator_name")
		.value("no_rejection", Robust_Regression::estimator_name::no_rejection)
		.value("tolerance_is_maximum_squared_error", Robust_Regression::estimator_name::tolerance_is_maximum_squared_error)
		.value("tolerance_multiplies_standard_deviation_of_error", Robust_Regression::estimator_name::tolerance_multiplies_standard_deviation_of_error)
		.value("tolerance_is_significance_in_Grubbs_test", Robust_Regression::estimator_name::tolerance_is_significance_in_Grubbs_test)
		.value("tolerance_is_decision_in_MAD_ESTIMATION", Robust_Regression::estimator_name::tolerance_is_decision_in_MAD_ESTIMATION)
		.value("tolerance_is_decision_in_S_ESTIMATION", Robust_Regression::estimator_name::tolerance_is_decision_in_S_ESTIMATION)
		.value("tolerance_is_decision_in_Q_ESTIMATION", Robust_Regression::estimator_name::tolerance_is_decision_in_Q_ESTIMATION)
		.value("tolerance_is_decision_in_T_ESTIMATION", Robust_Regression::estimator_name::tolerance_is_decision_in_T_ESTIMATION)
		.value("use_peirce_criterion", Robust_Regression::estimator_name::use_peirce_criterion)
		.value("tolerance_is_biweight_midvariance", Robust_Regression::estimator_name::tolerance_is_biweight_midvariance)
		.export_values()
		;



	py::class_< Robust_Regression::control, LossFunctions::errorfunction>(robustregression, "control")
		.def(py::init<>())
		.def_readwrite("outlier_tolerance", &Robust_Regression::control::outlier_tolerance)
		.def_readwrite("rejection_method", &Robust_Regression::control::rejection_method)
		.def_readwrite("stop_after_seconds", &Robust_Regression::control::stop_after_seconds)
		.def_readwrite("stop_after_numberofiterations_without_improvement", &Robust_Regression::control::stop_after_numberofiterations_without_improvement)
		.def_readwrite("maximum_number_of_outliers", &Robust_Regression::control::maximum_number_of_outliers)
		;

	py::class_< Robust_Regression::linear_algorithm_control,Robust_Regression::control>(robustregression, "linear_algorithm_control")
		.def(py::init<>())
		.def_readwrite("use_median_regression", &Robust_Regression::linear_algorithm_control::use_median_regression)
		.def_readwrite("tolerable_error", &Robust_Regression::linear_algorithm_control::tolerable_error)
		;

	py::class_< Robust_Regression::nonlinear_algorithm_control,	Robust_Regression::control, Non_Linear_Regression::control>(robustregression, "nonlinear_algorithm_control")
		.def(py::init<>())
		;

	py::class_ < Robust_Regression::lts_control>(robustregression, "lts_control")
		.def(py::init<>())
		.def_readwrite("use_ransac", &Robust_Regression::lts_control::use_ransac)
		.def_readwrite("workload_distributed_to_several_threads", &Robust_Regression::lts_control::workload_distributed_to_several_threads)
		;

	py::class_< Robust_Regression::modified_lts_control_linear, Robust_Regression::linear_algorithm_control, Robust_Regression::lts_control>(robustregression, "modified_lts_control_linear")
		.def(py::init<>())
		;
	py::class_< Robust_Regression::modified_lts_control_nonlinear, Robust_Regression::nonlinear_algorithm_control, Robust_Regression::lts_control>(robustregression, "modified_lts_control_nonlinear")
		.def(py::init<>())
		;


	robustregression.def("modified_lts_regression_linear", &Robust_Regression::modified_lts_regression_linear);
	robustregression.def("iterative_outlier_removal_regression_linear", &Robust_Regression::iterative_outlier_removal_regression_linear);
	robustregression.def("modified_lts_regression_nonlinear", &Robust_Regression::modified_lts_regression_nonlinear, py::call_guard<py::gil_scoped_release>());
	robustregression.def("iterative_outlier_removal_regression_nonlinear", &Robust_Regression::iterative_outlier_removal_regression_nonlinear);
	robustregression.def("linear_loss_function", &Robust_Regression::linear_loss_function);

	};

