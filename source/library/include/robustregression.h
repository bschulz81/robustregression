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

#include <stdint.h>
#include <vector>
#include "linearregression.h"
#include "matrixcode.h"
#include "nonlinearregression.h"
#include "robustregressionlib_exports.h"
#include "lossfunctions.h"
using namespace std;



namespace Robust_Regression
{
	//yields the indices of the points that were removed or used by the robust regression algorithms
	struct result :LossFunctions::error
	{
		vector<size_t> indices_of_removedpoints;
		vector<size_t> indices_of_used_points;
	};

	//stores the result of the robust linear regression algorithms
	struct linear_algorithm_result :Robust_Regression::result, Linear_Regression::result
	{};

	//stores the result of the robust nonlinear regression algorithms
	struct nonlinear_algorithm_result :Robust_Regression::result, Non_Linear_Regression::result
	{};

	//the names of the statistical estimators that are used to remove outliers.
	//one has a tolerance parameter which can be used to set a measure of what an outlier is with respect to a statistical estimator
	enum estimator_name
	{
		no_rejection, //no rejection takes place
		tolerance_is_maximum_error, //The tolerance parameter sets the maximally tolerable errorof a point,
	    //where error is computed by some metric,i.e. the absolute value of the residuals, the squared residuals
		// or Huber's loss function. Choosing this estimator is not recommended since it is not robust.

		tolerance_multiplies_standard_deviation_of_error,//a point with an error is an outlier
		//if |error-average(errors)|>tolerance*standard_deviation(errors). The errors
		//are computed by some metric, i.e. the absolute value of the residuals, the squared residuals or Huber's loss function.
		//It is not recommended to use this estimator since it is not robust

		tolerance_is_significance_in_Grubbs_test, // A point is
		// an outlier if it its error is determined as such by the Grubbs test, given the distribution of the errors,
		// with a significance defined the tolerance parameter. The errors are computed by some metric, 
		// i.e. the absolute value of the residuals, the squared residuals or Huber's loss function.

		tolerance_is_decision_in_MAD_ESTIMATION,// A point with an error is an outlier if |error-median(errors)|/MAD(errors)>tolerance
		// The errors are computed by some metrici.e. the absolute value of the residuals, the squared residuals or Huber's loss function.
		// This estimator is robust and recommended.

		tolerance_is_decision_in_S_ESTIMATION,// A point with an error is an outlier if |error-median(errors)|/S_estimator(errors)>tolerance
		// The errors are computed by some metrici.e. the absolute value of the residuals, the squared residuals or Huber's loss function.
		//  This estimator is robust and recommended.

		tolerance_is_decision_in_Q_ESTIMATION,// A point with an error is an outlier if |error-median(errors)|/Q_estimator(errors)>tolerance
		// The errors are computed by some metrici.e. the absolute value of the residuals, the squared residuals or Huber's loss function.
		//  This estimator is robust and recommended.

		tolerance_is_decision_in_T_ESTIMATION,// A point with an error is an outlier if |error-median(errors)|/T_estimator(errors)>tolerance
		// The errors are computed by some metrici.e. the absolute value of the residuals, the squared residuals or Huber's loss function.
		//  This estimator is robust and recommended.

		use_peirce_criterion,// If set, the Peirce criterion is used on the array of errors to establish whether a point with an error is
		//an outlier. The errors are computed by some metric, i.e. the absolute value of the residuals, the squared residuals or
		//  Huber's loss function.

		tolerance_is_biweight_midvariance,// A point with an error is an outlier if
		//|error-median(errors)|/Biweight_midvariance(errors)>tolerance. The errors are computed by some metrici.e. the absolute value
		//  of the residuals, the squared residuals or Huber's loss function.
		tolerance_is_interquartile_range // A point is an outlier its error is smaller than Q1-tolerance*Inter_Quartile_Range
		//or larger than Q2+tolerance*Inter_Quartile_Range The errors are computed by some metrici.e.the absolute value of the 
		// residuals, the squared residuals or Huber's loss function.
	};

	//defines the controlparameters common in all the robust regression algorithms
	//the outlier_tolerance parameter sets a scalar quantity, together with the statistical estimator given by rejection_method,
	//how outliers are to be rejected.then one has various parameters that set a time or an iteration number after which the
	//algorithm stops if no improvement has been made.
	//finally, the maximum number of outliers sets the maximum number of outliers which the robust algorithms can find. 
	// if set to zero, the algorithms will designate at maximum 30% of the input datapoints as outliers  
	struct control
	{
		double outlier_tolerance{ 3.0 };
		estimator_name rejection_method{ tolerance_is_decision_in_S_ESTIMATION };
		double stop_after_seconds{ 30 };
		size_t stop_after_numberofiterations_without_improvement{ 40000 };
		//if maximum_number_of_outliers is not set by the user, the system will use 30% of the points as maximum, per default...
		size_t maximum_number_of_outliers{ 0 };
	};

	//defines controlparameters for the linear robust algorithm. Whether median regression will be used inside the robust
	//algorithms, and a tolerable error, after which the algorithm stops. 
	struct linear_algorithm_control : Robust_Regression::control, LossFunctions::errorfunction
	{
		bool use_median_regression{ false };
		double tolerable_error{ DBL_EPSILON };
	};

	//defines controlparameters for the nonlinear robust algorithm
	struct nonlinear_algorithm_control : Robust_Regression::control, Non_Linear_Regression::control
	{
	};

	//defines controlparameters for the modified least trimmed squares algorithm.
	//one can set whether it should use a ransac or iterate deterministically through all start models
	//finally one has a parameter that sets a number of curve fits which are then distributed over several threads.
	struct lts_control
	{
		bool use_ransac{ false };
		size_t workload_distributed_to_several_threads{ 705432 };
	};

	//defines controlparameters for the linear modified lts algorithms
	struct modified_lts_control_linear:linear_algorithm_control, lts_control
	{
	};

	//defines controlparameters for the nonlinear modified lts algorithms
	struct modified_lts_control_nonlinear :nonlinear_algorithm_control, lts_control
	{
	};
	

	//Implements a modified least trimmed squares algorithm.
	//one starts with a minimal model of a given size. It consists of a combination of points from the data and has the size pointnumber-maximum_number_of_outliers
	//Then one makes a curve fit and looks whether the other points in the dataset would be outliers based on their residuals and statistical estimators.
	//If the points are not outliers, they are added to the model.
	//A curve fit with the enlarged model is done, and then one starts the process from another combination of pointnumber-maximum_number_of_outliers points
	//The best fit that was found is given as a result. This function is for linear regression.
	ROBUSTREGRESSION_API bool modified_lts_regression_linear(const valarray<double>& x, const valarray<double>& y,
		Robust_Regression::modified_lts_control_linear& controldata,
		Robust_Regression::linear_algorithm_result& result);

	//Implements an iterative outlier removal algorithm for linear regression.
	// One starts with a curve fit of all points. Then one looks at the data and estimates whether the residuals computed for certain points are outliers.
	// These points are then removed, and another fit is made. The process continues until some specified maximum number of outliers is removed or no
	// outliers are found anymore. 
	ROBUSTREGRESSION_API bool iterative_outlier_removal_regression_linear(const valarray<double>& x, const valarray<double>& y,
		Robust_Regression::linear_algorithm_control& controldata,
		Robust_Regression::linear_algorithm_result& result);

	//Implements a modified least trimmed squares algorithm.
	//one starts with a minimal model of a given size. It consists of a combination of points from the data and has the size pointnumber-maximum_number_of_outliers
	//Then one makes a curve fit and looks whether the other points in the dataset would be outliers. If not, they are added to the model.
	//A curve fit with the enlarged model is done, and then one starts the process from another combination of pointnumber-maximum_number_of_outliers points
	//The best fit that was found is given as a result. This function is for non-linear regression.
	ROBUSTREGRESSION_API bool modified_lts_regression_nonlinear(const valarray<double>& x, const valarray<double>& y,
		Non_Linear_Regression::initdata& init,
		Robust_Regression::modified_lts_control_nonlinear& controldata,
		Robust_Regression::nonlinear_algorithm_result& result);

	//Implements an iterative outlier removal algorithm for nonlinear regression.
	// One starts with a curve fit of all points. Then one looks at the data and estimates whether the residuals computed for certain points are outliers.
	// These points are then removed, and another fit is made. The process continues until some specified maximum number of outliers is removed or no
	// outliers are found anymore. 
	ROBUSTREGRESSION_API bool iterative_outlier_removal_regression_nonlinear(const valarray<double>& x, const valarray<double>& y,
		Non_Linear_Regression::initdata& init,
		Robust_Regression::nonlinear_algorithm_control& ctrl,
		Robust_Regression::nonlinear_algorithm_result& result);

	
	//Implements a loss function for a linear curve fit that computes the residuals for every point given some metric 
   // (squared residuals, Huber's loss function, absolute value), and the sum of these residuals
	ROBUSTREGRESSION_API double linear_loss_function(const valarray<double>& x, const valarray<double>& y,
		LossFunctions::errorfunction& ctrl, Robust_Regression::linear_algorithm_result& err);
}
