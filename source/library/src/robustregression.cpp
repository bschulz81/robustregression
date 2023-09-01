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


#include <vector>
#include <algorithm>
#include <iostream>
#include <random>  
#include <valarray>
#include <atomic>
#include <unordered_set>
#include <complex>
#include <omp.h>
#include <chrono>

#include "statisticfunctions.h"
#include "lossfunctions.h"
#include "linearregression.h"

#include "robustregression.h"

using namespace std;

#if __cplusplus == 201703L
	#include <mutex>
	#include <execution>
#endif



#ifdef GNUCOMPILER
#define _inline inline
#endif

#ifdef CLANGCOMPILER
#define _inline inline
#endif


using namespace std;

struct maybe_inliner
{
	size_t point;
	double error;
};



typedef bool(*linreg)(const valarray<double>& x, const valarray<double>& y, Linear_Regression::result& res);

typedef bool(*nonlinreg)(const valarray<double>& X, const valarray<double>& Y, Non_Linear_Regression::initdata &init,
																				Non_Linear_Regression::control& control,
																				Non_Linear_Regression::result& res);

inline bool isoutlier(double err, Robust_Regression::estimator_name rejection_method, double tolerance, double average_or_median, double estimators);

inline void computew1w2estimator(valarray<double>& err, size_t pointnumber, double& est1, double& est2, Robust_Regression::estimator_name estimator_name);

inline void helperfunction_findmodel(valarray<double>& err, valarray<bool>* usedpoint,  Robust_Regression::control& ctrl);


inline void findmodel_linear(const valarray<double>& x, const valarray<double>& y, valarray<bool>* usedpoint,
	linreg regr,  Robust_Regression::control& ctrl);

inline bool findmodel_non_linear(const valarray<double>& x, const valarray<double>& y, valarray<bool>* usedpoint,
	nonlinreg regr, Non_Linear_Regression::initdata& init, Robust_Regression::nonlinear_algorithm_control& ctrl);


inline bool helperfunction_least_trimmed(const valarray<double>& x, const valarray<double>& y, valarray<bool>* indices, valarray<bool>* indices2,
	linreg regr,
	Robust_Regression::linear_algorithm_result* result_lin,
	Robust_Regression::linear_algorithm_control* controldata_lin,
	nonlinreg nonlinreg,
	Non_Linear_Regression::initdata* init_nonlin,
	Robust_Regression::nonlinear_algorithm_control* controldata_nonlin,
	Robust_Regression::nonlinear_algorithm_result* result_nonlin
);


inline bool checkdata_robustmethods(const valarray<double>& x, const valarray<double>& y,
	Robust_Regression::control& controldata);

inline bool checkdata_nonlinear_methods(const valarray<double>& x, const valarray<double>& y,
	Robust_Regression::nonlinear_algorithm_control& controldata, Non_Linear_Regression::initdata& init);

inline void fill_robustdata(const valarray<double>& x, const valarray<double>& y, valarray<bool>& indices,
	Robust_Regression::result& result,
	Robust_Regression::control& controldata);




inline bool isoutlier(double err, Robust_Regression::estimator_name rejection_method, double tolerance , double w1, double w2 )
{
	switch (rejection_method)
	{
	case Robust_Regression::tolerance_is_maximum_squared_error:
	{
		double G = err * err;
		if (G > Statisticfunctions::fabs(tolerance))
		{
			return true;
		}
		break;
	}
	case Robust_Regression::use_peirce_criterion:
	{
		double G = err * err;
		if (G > w1 * tolerance)
		{
			return true;
		}
		break;
	}
	case::Robust_Regression::tolerance_is_interquartile_range:
	{
		bool b =err < (w1 - tolerance * (w2 - w1)) ?  true : err > (w2 + tolerance * (w2 - w1)) ? true : false;
		return b;
		break;
	}
	default:
	{
		double G = Statisticfunctions::fabs((err - w1) / w2);
		if (G > tolerance)
		{
			return true;
		}
		break;
	}
	}
	return false;
}


inline void computew1w2estimator(valarray<double>& err, size_t pointnumber,double & w1,double& w2, Robust_Regression::estimator_name estimator_name)
{
	switch (estimator_name)
	{
	case Robust_Regression::tolerance_is_maximum_squared_error:
	{
		break;
	}

	case Robust_Regression::tolerance_multiplies_standard_deviation_of_error:
	{
		//falls through and computes stdev and average
	}
	case Robust_Regression::tolerance_is_significance_in_Grubbs_test:
	{
		//falls through and computes stdev and average
		w2 = Statisticfunctions::stdeviation(err); 
	}
	case Robust_Regression::use_peirce_criterion:
	{
		w1 = Statisticfunctions::average(err);
		break;
	}
	case Robust_Regression::tolerance_is_decision_in_MAD_ESTIMATION:
	{
		w1 = Statisticfunctions::median(err);
		w2 = Statisticfunctions::MAD_estimator(err, w1);
		break;
	}
	case Robust_Regression::tolerance_is_biweight_midvariance:
	{
		w1 = Statisticfunctions::median(err);
		w2 = Statisticfunctions::onestepbiweightmidvariance(err, w1);
		break;
	}
	case Robust_Regression::tolerance_is_decision_in_Q_ESTIMATION:
	{
		w1 = Statisticfunctions::median(err);
		w2 = Statisticfunctions::Q_estimator(err);
		break;
	}
	case Robust_Regression::tolerance_is_decision_in_S_ESTIMATION:
	{
		w1 = Statisticfunctions::median(err);
		w2 = Statisticfunctions::S_estimator(err);
		break;
	}
	case Robust_Regression::tolerance_is_decision_in_T_ESTIMATION:
	{
		w1 = Statisticfunctions::median(err);
		w2 = Statisticfunctions::T_estimator(err );
		break;
	}
	case::Robust_Regression::tolerance_is_interquartile_range:
		w1 = Statisticfunctions::Q1(err);
		w2 = Statisticfunctions::Q3(err);
	}
}

inline void helperfunction_findmodel(valarray<double>&err, valarray<bool>* usedpoint, Robust_Regression::control& ctrl)
{
	size_t pointnumber = err.size();

	vector<maybe_inliner> mp;
	mp.reserve(pointnumber);


	for (size_t p = 0; p < pointnumber; p++)
	{

		if (!(*usedpoint)[p])
		{
			maybe_inliner o = { p, (err)[p] };
			mp.push_back(o);
		}
	}

	if (mp.size() > 0)
	{
		double estimate1 = 0, estimate2 = 0;
		computew1w2estimator(err, pointnumber, estimate1, estimate2, ctrl.rejection_method);

		for (size_t j = 0; j < mp.size(); j++)
		{

			if (isoutlier(mp[j].error, ctrl.rejection_method, ctrl.outlier_tolerance, estimate1,estimate2 ))
				(*usedpoint)[mp[j].point] = false;
			else
				(*usedpoint)[mp[j].point] = true;
		}
	}
}



inline void findmodel_linear(const valarray<double>& x, const valarray<double>& y, valarray<bool>* usedpoint,
	linreg regr,  Robust_Regression::control&ctrl)
{

	size_t pointnumber = x.size();
	valarray<double> x1 = x[*usedpoint];
	valarray<double> y1 = y[*usedpoint];


	Robust_Regression::linear_algorithm_result res{};

	regr(x1, y1, res);


	linear_loss_function(x, y,ctrl, res );

	helperfunction_findmodel(res.errorarray, usedpoint,ctrl);
}



inline bool findmodel_non_linear(const valarray<double>& x, const valarray<double>& y, valarray<bool>* usedpoint,
	nonlinreg regr, Non_Linear_Regression::initdata &init, Robust_Regression::nonlinear_algorithm_control& ctrl)
{
	//makes a fit with the initial usedpoint array as a mask
	size_t pointnumber = x.size();
	valarray<double> x1 = x[*usedpoint];
	valarray<double> y1 = y[*usedpoint];


	Robust_Regression::nonlinear_algorithm_result res{};


	regr(x1, y1, init, ctrl, res);

	
	Non_Linear_Regression::nonlinear_loss_function(init.f, x, res.beta, y, ctrl, res);
	helperfunction_findmodel(res.errorarray,usedpoint, ctrl);

	return true;
}

void helperfunction_last_trimmed2(const valarray<double>& x, const valarray<double>& y, valarray<bool>* indices, valarray<bool>* indices2, linreg regr,
	Robust_Regression::linear_algorithm_result* result_lin,
	Robust_Regression::modified_lts_control_linear* controldata_lin,
	nonlinreg nonlinreg,
	Non_Linear_Regression::initdata* init_nonlin,
	Robust_Regression::modified_lts_control_nonlinear* controldata_nonlin,
	Robust_Regression::nonlinear_algorithm_result* result_nonlin, size_t numbercomp, size_t pointnumber,size_t number_of_attempts,bool random, std::mt19937 * urng,size_t &counter_init)
{
#if __cplusplus == 201703L 
	std::mutex mtx;
	std::atomic<size_t> counter1{ counter_init };
#else
	size_t counter1 = counter_init;
#endif

	

	std::unordered_set<std::vector<bool>> helper3;
	vector<valarray<bool>> arr;



	if (random == true)
	{
		arr.reserve(numbercomp);
		std::vector<bool> helper2;
		for (size_t i = 0; i < number_of_attempts; i++)
		{
			std::shuffle(std::begin(*indices), std::end(*indices),*urng);
			helper2.assign(std::begin(*indices), std::end(*indices));
			bool b2 = helper3.insert(helper2).second;
			if (b2 == true)
				arr.push_back(*indices);
		}
	}
	else
	{
		arr.resize(numbercomp);

		for (size_t i = 0; i < numbercomp; i++)
		{
			arr[i] = *indices;
			std::next_permutation(std::begin(*indices), std::end(*indices));
		}
	}

	vector<valarray<bool>> arr2;
	arr2.reserve(numbercomp);

#if __cplusplus == 201703L && !defined(MACOSX)
	std::for_each(std::execution::par, std::begin(arr), std::end(arr), [&](valarray<bool>& arri)
		{
			if (controldata_lin != NULL)
				findmodel_linear(x, y, &arri, regr, *controldata_lin);
			else
				findmodel_non_linear(x, y, &arri, nonlinreg, *init_nonlin, *controldata_nonlin);
			std::vector<bool> helper(pointnumber);
			helper.assign(std::begin(arri), std::end(arri));
			mtx.lock();
			bool b2 = helper3.insert(helper).second;
			if (b2)
				arr2.push_back(arri);
			mtx.unlock();
			counter1++;
		});


	std::for_each(std::execution::par, std::begin(arr2), std::end(arr2), [&](valarray<bool>& arri)
		{
				valarray<double> xnew = x[arri];
				valarray<double> ynew = y[arri];
				if (controldata_lin != NULL)
				{
					Robust_Regression::linear_algorithm_result res{};
					regr(xnew, ynew, res);
					linear_loss_function(xnew, ynew, *controldata_lin, res);
					mtx.lock();
					if (res.main_error < result_lin->main_error)
					{
						result_lin->main_error = res.main_error;
						result_lin->main_slope = res.main_slope;
						result_lin->main_intercept = res.main_intercept;
						*indices2 = arri;
						if(counter1>0) counter1--;
					}
					mtx.unlock();
				}
				else
				{
					Robust_Regression::nonlinear_algorithm_result res{};
					nonlinreg(xnew, ynew, *init_nonlin, *controldata_nonlin, res);
					Non_Linear_Regression::nonlinear_loss_function(init_nonlin->f, xnew, res.beta, ynew, *controldata_nonlin, res);
					mtx.lock();
					if (res.main_error < result_nonlin->main_error)
					{
						result_nonlin->beta = res.beta;
						result_nonlin->main_error = res.main_error;
						*indices2 = arri;
						if (counter1 > 0) counter1--;
					}
					mtx.unlock();
				}
		});
#else
#pragma omp parallel for
	for (long i = 0; i < arr.size(); i++)
	{
		if (controldata_lin != NULL)
			findmodel_linear(x, y, &arr[i], regr, est, *controldata_lin);
		else
			findmodel_non_linear(x, y, &arr[i], nonlinreg, est, *init_nonlin, *controldata_nonlin);
		//for each model we tried to find increase counter1 by 1
#pragma omp atomic
		counter1++;

		std::vector<bool> helper(pointnumber);
		helper.assign(std::begin(arr[i]), std::end(arr[i]));
#pragma omp critical
		{
			bool b2 = helper3.insert(helper).second;
			if (b2)
				arr2.push_back(arr[i]);
		}
	}

#pragma omp parallel for
	for (long i = 0; i < arr2.size(); i++)
	{
		valarray<double> xnew = x[arr2[i]];
		valarray<double> ynew = y[arr2[i]];
		if (controldata_lin != NULL)
		{
			Robust_Regression::linear_algorithm_result res{};
			regr(xnew, ynew, res);
			linear_loss_function(xnew, ynew, *controldata_lin, res);
#pragma omp critical
			{
				if (res.main_error < result_lin->main_error)
				{
					result_lin->main_error = res.main_error;
					result_lin->main_slope = res.main_slope;
					result_lin->main_intercept = res.main_intercept;
					*indices2 = arr2[i];
					if (counter1 > 0) counter1--;
				}
			}
		}
		else
		{
			Robust_Regression::nonlinear_algorithm_result res{};
			nonlinreg(xnew, ynew, *init_nonlin, *controldata_nonlin, res);
			Non_Linear_Regression::nonlinear_loss_function(init_nonlin->f, xnew, res.beta, ynew, *controldata_nonlin, res);
#pragma omp critical
			{
				if (res.main_error < result_nonlin->main_error)
				{
					result_nonlin->beta = res.beta;
					result_nonlin->main_error = res.main_error;
					*indices2 = arr2[i];
					if (counter1 > 0) counter1--;
				}
			}
		}
	}

#endif

	counter_init = counter1;

}


bool check_terminate_loop(size_t counter1, std::chrono::steady_clock::time_point start,
	Robust_Regression::linear_algorithm_control* controldata_lin, 
	Robust_Regression::nonlinear_algorithm_control* controldata_nonlin,
	Robust_Regression::nonlinear_algorithm_result* result_nonlin,
	Robust_Regression::linear_algorithm_result* result_lin)
{
			if (controldata_lin != NULL)
			{
				if (controldata_lin->tolerable_error >= result_lin->main_error)
				{
					return true;
				}
			}
			else
			{
				if (controldata_nonlin->tolerable_error >= result_nonlin->main_error)
				{
					return true;
				}
			}
			size_t s0;

			if (controldata_lin != NULL)
			{
				if (controldata_lin->tolerable_error >= result_lin->main_error)
				{
					return true;
				}
				s0 = controldata_lin->stop_after_numberofiterations_without_improvement;

			}
			else
			{
				if (controldata_nonlin->tolerable_error >= result_nonlin->main_error)
					return true;
				s0 = controldata_nonlin->stop_after_numberofiterations_without_improvement;
			}
			if (counter1 >= s0)
			{
				return true;
			}

			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - start;
			long seconds = elapsed_seconds.count();
			//after the time specified by the user has passed, stop the algorithm
			double s1;
			if (controldata_lin != NULL)
				s1 = controldata_lin->stop_after_seconds;
			else
				s1 = controldata_nonlin->stop_after_seconds;

			if (seconds > Statisticfunctions::fabs(s1))
				return true;

			return false;
}



inline bool helperfunction_least_trimmed(const valarray<double>& x, const valarray<double>& y, valarray<bool>* indices, valarray<bool>* indices2,	linreg regr,
	Robust_Regression::linear_algorithm_result* result_lin,
	Robust_Regression::modified_lts_control_linear* controldata_lin,
	nonlinreg nonlinreg,
	Non_Linear_Regression::initdata* init_nonlin,
	Robust_Regression::modified_lts_control_nonlinear* controldata_nonlin,
	Robust_Regression::nonlinear_algorithm_result* result_nonlin)
{


	//how many computations do we have to make based on the minimum fit model size
	size_t pointnumber = x.size();


	size_t numbercomp = 0;
	try {
		if (controldata_lin != NULL)
			numbercomp = Statisticfunctions::binomial(pointnumber, controldata_lin->maximum_number_of_outliers);
		else
			numbercomp = Statisticfunctions::binomial(pointnumber, controldata_nonlin->maximum_number_of_outliers);
	}
	catch (...)
	{
		cout << "An exception occurred. Binomial from the pointnumber and maximum number of outliers could not be computed. Probably too large";
		return false;
	}




	//set some number of computations which should be distributed on several processors 
	bool useransac;
	size_t workload_in__several_threads;

	//check if the supplied stop_after_numberofiterations_without_improvement was too small and set the error to max.
	if (controldata_lin != NULL)
	{
		result_lin->main_error = DBL_MAX;
		useransac = controldata_lin->use_ransac;
		workload_in__several_threads = controldata_lin->workload_distributed_to_several_threads;
		if (controldata_lin->stop_after_numberofiterations_without_improvement < workload_in__several_threads)
			controldata_lin->stop_after_numberofiterations_without_improvement = workload_in__several_threads;

		
	}
	else
	{

		result_nonlin->main_error = DBL_MAX;
		workload_in__several_threads = controldata_nonlin->workload_distributed_to_several_threads;
		useransac = controldata_nonlin->use_ransac;
		if (controldata_nonlin->stop_after_numberofiterations_without_improvement < workload_in__several_threads)
			controldata_nonlin->stop_after_numberofiterations_without_improvement = workload_in__several_threads;


	}

	

	if ((numbercomp <= workload_in__several_threads) && (useransac == false))
	{
		size_t counter_false_attempts = 0;
		std::chrono::steady_clock::time_point start  = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < numbercomp; i++)
		{
			helperfunction_last_trimmed2(
				x, y, indices, indices2, regr,	result_lin,		controldata_lin,
				nonlinreg,	init_nonlin,controldata_nonlin,result_nonlin,
				numbercomp, pointnumber, workload_in__several_threads, false, NULL, counter_false_attempts);
			if (check_terminate_loop(counter_false_attempts, start, controldata_lin, controldata_nonlin, result_nonlin, result_lin))
				break;
		}
		
	}

	else if ((numbercomp > workload_in__several_threads) && (useransac==false))
	{

		size_t p = (size_t)numbercomp / workload_in__several_threads;

		bool endreached=true;
		size_t counter_false_attempts = 0;
		std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < numbercomp; i++)
		{
			helperfunction_last_trimmed2(
				x, y, indices, indices2, regr, result_lin, controldata_lin,
				nonlinreg, init_nonlin, controldata_nonlin, result_nonlin,
				numbercomp, pointnumber, workload_in__several_threads, false, NULL,counter_false_attempts);
			if (check_terminate_loop(counter_false_attempts, start, controldata_lin, controldata_nonlin, result_nonlin, result_lin))
			{
				endreached=false;
				break;
			}
		}
		if (endreached == true)
		{
			size_t k = (numbercomp % workload_in__several_threads) + 1;
			std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
			for (size_t i = 0; i < k; i++)
			{
				helperfunction_last_trimmed2(
					x, y, indices, indices2, regr, result_lin, controldata_lin,
					nonlinreg, init_nonlin, controldata_nonlin, result_nonlin,
					k, pointnumber, workload_in__several_threads, false, NULL, counter_false_attempts);

				if (check_terminate_loop(counter_false_attempts, start, controldata_lin, controldata_nonlin, result_nonlin, result_lin))
				{
					endreached = false;
					break;
				}
			}
		}
	}
	else
	{
		std::random_device rng;
		std::mt19937 urng(rng());
		size_t counter_false_attempts = 0;
		std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
		do
		{
			helperfunction_last_trimmed2(
				x, y, indices, indices2, regr, result_lin, controldata_lin,
				nonlinreg, init_nonlin, controldata_nonlin, result_nonlin,
				numbercomp, pointnumber, workload_in__several_threads, true, &urng, counter_false_attempts);

		} while (!check_terminate_loop(counter_false_attempts, start, controldata_lin, controldata_nonlin, result_nonlin, result_lin));
	}



	if (result_lin != NULL)
	{
		if (result_lin->main_error == DBL_MAX)
		{
			return false;
		}
	}
	else
	{
		if (result_nonlin->main_error == DBL_MAX)
		{
			return false;
		}

	}
	return true;



}


inline bool checkdata_robustmethods(const valarray<double>& x, const valarray<double>& y,
	Robust_Regression::control& controldata)
{

	if (x.size() < 4)
	{
		return false;
	}

	if (y.size() < x.size())
	{
		return false;
	}
	

	size_t pointnumber = x.size();


	if (controldata.rejection_method == Robust_Regression::no_rejection)
	{
		controldata.maximum_number_of_outliers = 0;
	}
	else
	{
		//if no maximum number of outliers was specified, use 30% of the pointnumber per default
		if (controldata.maximum_number_of_outliers == 0)
		{
			controldata.maximum_number_of_outliers = pointnumber * 0.3;
		}
	}

	size_t minimummodelsize = pointnumber - controldata.maximum_number_of_outliers;
	//ensure, we fit at least 4 points
	if (minimummodelsize < 4)
	{
		minimummodelsize = 4;
		controldata.maximum_number_of_outliers = pointnumber - minimummodelsize;
	}

	if (controldata.rejection_method == Robust_Regression::tolerance_is_significance_in_Grubbs_test)
	{
		if (controldata.outlier_tolerance >= 1.0)
		{
			controldata.rejection_method = Robust_Regression::no_rejection;
			controldata.maximum_number_of_outliers = 0;
		}
	}

	if (controldata.huberslossfunction_border <= 0)
		controldata.lossfunction = LossFunctions::squaredresidual;

	if (controldata.outlier_tolerance == 0)
	{
		controldata.maximum_number_of_outliers = 0;
		controldata.rejection_method = Robust_Regression::no_rejection;
	}
	return true;
}

inline bool checkdata_nonlinear_methods(const valarray<double>& x, const valarray<double>& y,
	Robust_Regression::nonlinear_algorithm_control& controldata, Non_Linear_Regression::initdata& init)
{
	if (checkdata_robustmethods(x, y, controldata) == false)
		return false;
	if (init.f == NULL)
		return false;
	if (controldata.lambda < 0)
		controldata.lambda = 4.0;
	if (controldata.increment < 0)
		controldata.increment = 1.5;
	if (controldata.decrement < 0)
		controldata.decrement = 5;
	if (controldata.precision < 0)
		controldata.precision = 0.01;
	if (controldata.h < 0)
		controldata.h = 0.01;
	if (controldata.stop_nonlinear_curve_fitting_after_iterations < 0)
		controldata.stop_nonlinear_curve_fitting_after_iterations = 1;
	if (controldata.stop_nonlinear_curve_fitting_after_seconds < 0)
		controldata.stop_nonlinear_curve_fitting_after_seconds = 1;
	if (controldata.tolerable_error < DBL_EPSILON)
		controldata.tolerable_error = DBL_EPSILON;
	return true;
}


inline void fill_robustdata(const valarray<double>& x, const valarray<double>& y, valarray<bool>& indices,
	Robust_Regression::result& result,
	Robust_Regression::control& controldata) 
{
	size_t pointnumber = indices.size();


		result.indices_of_used_points.clear();
		result.indices_of_removedpoints.clear();

		for (size_t k = 0; k < pointnumber; k++)
		{
			if ((indices)[k])
			{
				result.indices_of_used_points.push_back(k);
			}
			else
			{
				result.indices_of_removedpoints.push_back(k);
			}
		}
	}

ROBUSTREGRESSION_API inline  bool Robust_Regression::iterative_outlier_removal_regression_linear(const valarray<double>& x, const valarray<double>& y,
	Robust_Regression::linear_algorithm_control& controldata, Robust_Regression::linear_algorithm_result& result)
{

	if (!checkdata_robustmethods(x, y, controldata))
		return false;

	size_t pointnumber = x.size();
	valarray<bool> indices(true, pointnumber);
	if (controldata.tolerable_error < DBL_EPSILON)
		controldata.tolerable_error = DBL_EPSILON;

	if (controldata.rejection_method == Robust_Regression::no_rejection)
	{
		if (controldata.use_median_regression)
		{
			Linear_Regression::median_linear_regression(x, y, result);
		}
		else
		{
			Linear_Regression::linear_regression(x, y, result);
		}

		linear_loss_function(x, y, controldata,result);
		fill_robustdata(x, y, indices, result, controldata);
		return true;
	}

	if (controldata.rejection_method == Robust_Regression::tolerance_is_significance_in_Grubbs_test)
	{
		if (controldata.outlier_tolerance >= 1.0)
		{
			if (controldata.use_median_regression)
			{
				Linear_Regression::median_linear_regression(x, y, result);
			}
			else
			{
				Linear_Regression::linear_regression(x, y, result);
			}
			linear_loss_function(x, y,controldata,result);
			fill_robustdata(x, y, indices, result, controldata);
			return true;
		}
	}


	valarray<double>xv(pointnumber);
	valarray<double>yv(pointnumber);
	xv = x;
	yv = y;

	size_t counter = 0;
	size_t arraylength0 = 0, arraylengthnew;


	valarray<double>* xv1 = &xv;
	valarray<double>* yv1 = &yv;

	linreg regr;
	if (controldata.use_median_regression)
	{
		regr = &(Linear_Regression::median_linear_regression);
	}
	else
	{
		regr = &(Linear_Regression::linear_regression);
	}

	result.main_error = DBL_MAX;
	do
	{
		arraylengthnew = (*xv1).size();
		Robust_Regression::linear_algorithm_result res{};

		regr(*xv1, *yv1, res);

		linear_loss_function(*xv1, *yv1, controldata,res);

		if (res.main_error == DBL_MAX)
		{
			break;
		}

		if (res.main_error < result.main_error)
		{
			result.main_error = res.main_error;
			result.main_intercept = res.main_intercept;
			result.main_slope = res.main_slope;
		}
		else
		{
			counter++;
		}
		if (result.main_error <= controldata.tolerable_error)
			break;
		
		if (arraylengthnew <= 4)
			break;
		size_t dist = arraylengthnew - arraylength0;
		if (dist == 0)
			break;

		if (x.size() - arraylengthnew >= controldata.maximum_number_of_outliers)
			break;

		if (counter > controldata.stop_after_numberofiterations_without_improvement)
			break;

		double estimate1 = 0, estimate2 = 0;

		computew1w2estimator(res.errorarray, res.errorarray.size(), estimate1, estimate2, controldata.rejection_method);

		for (size_t i = 0; i < res.errorarray.size(); i++)
		{
			if ((isoutlier(res.errorarray[i], controldata.rejection_method, controldata.outlier_tolerance, estimate1, estimate2)) && indices[i] == true)
				indices[i] = false;
		}

		arraylength0 = arraylengthnew;

		valarray<double>xv2 = (*xv1)[indices];
		valarray<double>yv2 = (*yv1)[indices];
		res.errorarray = res.errorarray[indices];
		*xv1 = xv2;
		*yv1 = yv2;
	} while (true);

		double mainerr = result.main_error;
		linear_loss_function(x, y, controldata,result);
		//restore the old error, which was computed without outliers.
		result.main_error = mainerr;

	fill_robustdata(x, y, indices, result, controldata);
	return true;
}



ROBUSTREGRESSION_API inline bool  Robust_Regression::modified_lts_regression_linear(const valarray<double>& x, const valarray<double>& y,
	Robust_Regression::modified_lts_control_linear& controldata,
	Robust_Regression::linear_algorithm_result& result)
{


	size_t pointnumber = x.size();
	if (!checkdata_robustmethods(x, y, controldata))
		return false;

	if (controldata.tolerable_error < DBL_EPSILON)
		controldata.tolerable_error = DBL_EPSILON;


	valarray<bool> indices(pointnumber);
	valarray<bool> indices2(pointnumber);

	for (size_t i = 0; i < pointnumber; i++)
	{
		if (i >= controldata.maximum_number_of_outliers)
		{
			indices[i] = true;
		}
		else
		{
			indices[i] = false;
		}
	}

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(std::begin(indices), std::end(indices), g);

	linreg regr;
	if (controldata.use_median_regression)
	{
		regr = &(Linear_Regression::median_linear_regression);
	}
	else
	{
		regr = &(Linear_Regression::linear_regression);
	}
		


	
	if (controldata.maximum_number_of_outliers <= 0)
	{

		regr(x, y, result);
		linear_loss_function(x, y, controldata, result);
		fill_robustdata(x, y, indices, result, controldata);
		return true;
	}
	else
	{

		helperfunction_least_trimmed(x, y, &indices, &indices2, regr, &result, &controldata, NULL, NULL, NULL, NULL);
		fill_robustdata(x, y, indices2, result, controldata);
		return true;
	}
}



ROBUSTREGRESSION_API inline bool Robust_Regression::modified_lts_regression_nonlinear(const valarray<double>& x, const valarray<double>& y,
	Non_Linear_Regression::initdata& init,
	Robust_Regression::modified_lts_control_nonlinear&controldata,
	Robust_Regression::nonlinear_algorithm_result& result)
{
	
	size_t pointnumber = x.size();
	if (!checkdata_nonlinear_methods(x, y,controldata,init))
		return false;

	valarray<bool> indices(pointnumber);
	valarray<bool> indices2(pointnumber);

	for (size_t i = 0; i < pointnumber; i++)
	{
		if (i >= controldata.maximum_number_of_outliers)
		{
			indices[i] = true;
		}
		else
		{
			indices[i] = false;
		}
			
	}

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(std::begin(indices), std::end(indices), g);

	nonlinreg regr = (Non_Linear_Regression::non_linear_regression);


	if (controldata.maximum_number_of_outliers <= 0)
	{
		if (regr(x, y, init, controldata, result) == false)
			return false;
		Non_Linear_Regression:nonlinear_loss_function(init.f, x, result.beta, y,controldata, result);
		fill_robustdata(x, y,indices, result, controldata);
		return true;
	}
	else
	{
		helperfunction_least_trimmed(x, y, &indices, &indices2,NULL,NULL,NULL,regr,&init, &controldata,&result);
		fill_robustdata(x, y, indices2, result, controldata);
		return true;
	}
}


ROBUSTREGRESSION_API inline  bool Robust_Regression::iterative_outlier_removal_regression_nonlinear(const valarray<double>& x, const valarray<double>& y,
	Non_Linear_Regression::initdata& init,
	Robust_Regression::nonlinear_algorithm_control& ctrl,
	Robust_Regression::nonlinear_algorithm_result& result)
{
	
	if (!checkdata_nonlinear_methods(x, y, ctrl, init))
		return false;

	size_t pointnumber = x.size();
	valarray<bool> indices(true, pointnumber);

	if (ctrl.rejection_method == Robust_Regression::no_rejection)
	{
		if (Non_Linear_Regression::non_linear_regression(x, y, init, ctrl, result))
		{
			Non_Linear_Regression::nonlinear_loss_function(init.f, x, result.beta, y, ctrl, result);
			fill_robustdata(x, y, indices, result, ctrl);
			return true;
		}
		else
			return false;
	}


	if (ctrl.rejection_method == Robust_Regression::tolerance_is_significance_in_Grubbs_test)
	{
		if (ctrl.outlier_tolerance >= 1.0)
		{
			if (Non_Linear_Regression::non_linear_regression(x, y, init, ctrl, result))
			{
				Non_Linear_Regression::nonlinear_loss_function(init.f, x, result.beta, y, ctrl, result);
				fill_robustdata(x, y, indices, result, ctrl);
				return true;
			}
			else
				return false;
		}
	}



	valarray<double>xv(pointnumber);
	valarray<double>yv(pointnumber);
	xv = x;
	yv = y;

	size_t counter = 0;
	size_t arraylength0 = 0, arraylengthnew;


	valarray<double>* xv1 = &xv;
	valarray<double>* yv1 = &yv;

	nonlinreg regr = &(Non_Linear_Regression::non_linear_regression);

	result.main_error = DBL_MAX;
	result.beta = init.initialguess;
	do
	{
		arraylengthnew = (*xv1).size();
			
		Robust_Regression::nonlinear_algorithm_result res{};

		regr(*xv1, *yv1, init, ctrl, res);

		Non_Linear_Regression::nonlinear_loss_function(init.f,*xv1,res.beta,*yv1, ctrl, res);
		valarray<double> tmp = res.beta - result.beta;

		if ((tmp * tmp).sum() < ctrl.precision)
			break;

		if (res.main_error == DBL_MAX)
		{
			break;
		}

		if (res.main_error < result.main_error)
		{
			result.main_error = res.main_error;
			result.beta = res.beta;
		}
		else
		{
			counter++;
		}

		if (result.main_error <= ctrl.tolerable_error)
			break;

		if (arraylengthnew < 4)
			break;
		size_t dist = arraylengthnew - arraylength0;
		if (dist == 0)
			break;

		if (x.size()-arraylengthnew >= ctrl.maximum_number_of_outliers)
	    	break;

		if (counter > ctrl.stop_after_numberofiterations_without_improvement)
			break;



		double estimate1 = 0, estimate2 = 0;
		computew1w2estimator(res.errorarray, res.errorarray.size(), estimate1, estimate2, ctrl.rejection_method);
		for (size_t i = 0; i < res.errorarray.size(); i++)
		{
			if ((isoutlier(res.errorarray[i], ctrl.rejection_method, ctrl.outlier_tolerance, estimate1, estimate2)) && (indices[i] == true))
				indices[i] = false;	
		}

		arraylength0 = arraylengthnew;

		valarray<double>xv2 = (*xv1)[indices];
		valarray<double>yv2 = (*yv1)[indices];
		res.errorarray = res.errorarray[indices];
		*xv1 = xv2;
		*yv1 = yv2;
	} while (true);


	double mainerr = result.main_error;
	Non_Linear_Regression::nonlinear_loss_function(init.f, x, result.beta,y, ctrl, result);
		//restore the old error, which was computed without outliers.
	result.main_error = mainerr;

	fill_robustdata(x, y, indices, result, ctrl);

	return true;
}




ROBUSTREGRESSION_API inline  double  Robust_Regression::linear_loss_function(const valarray<double>& x, const valarray<double>& y,
	LossFunctions::errorfunction& ctrl,Robust_Regression::linear_algorithm_result& err)
{
	size_t pointnumber = x.size();
	err.errorarray.resize(pointnumber);
	double error = 0;
	err.main_error = 0;
	for (size_t p = 0; p < pointnumber; p++)
	{
		double z = err.main_slope * x[p] + err.main_intercept - y[p];
		switch (ctrl.lossfunction)
		{
		case LossFunctions::huberlossfunction:
		{
			error = (Statisticfunctions::fabs(z) <= ctrl.huberslossfunction_border) ? 0.5 * z * z / pointnumber : ctrl.huberslossfunction_border * (Statisticfunctions::fabs(z) - 0.5 * ctrl.huberslossfunction_border) / pointnumber;
			err.main_error += error;
			err.errorarray[p] = error;

			break;
		}

		case LossFunctions::squaredresidual:
		{
			error = z * z / pointnumber;
			err.main_error += error;
			err.errorarray[p] = error;
			break;
		}
		case LossFunctions::absolutevalue:
		{
			error = Statisticfunctions::fabs(z) / pointnumber;
			err.main_error += error;
			err.errorarray[p] = error;
			break;
		}
		}
	}
	return err.main_error;
}
