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

#include<valarray>
#include <functional>
#include <cfloat>
using namespace std;

namespace  LossFunctions
{

	//returns the error of the fit. It consists of an array with the residual error between every datapoint and the fitted function
	// and the main error, which is the sum of a certain metric, either
	// the square the residuals, 
	// ubers loss function of each residual, 
	// or the absolute value of each residual
	// of the points that were used in the fit.
	struct error
	{
		valarray<double> errorarray;
		double main_error{ DBL_MAX };
	};




	//errorfunction_name specifies the name of the metric that determines how the main error is computed.
	// 
	// squaredresidual sets the errorfunction per point to (y[i]-f(X[i]))^2/pointnumber, where pointnumber is the number of points used in the fit.
	// The entire loss is the sum of these errors.
	// Note that we divide by the number of points, otherwise, one would have a reduction of the error if we always reduce the pointsize.
	// 
	// absolutevalue sets the errorfunction per point to |y[i]-f(X[i])|/pointnumber, where pointnumber is the number of points used in the fit. 
	// The entire loss is the sum of these errors.
	// Note that we divide by the number of points, otherwise, one would have a reduction of the error if we always reduce the pointsize.
	// 
	// if Huber's loss function is used, one should set the parameter, called huberslossfunction_border which is called delta
	// in https://en.wikipedia.org/wiki/Huber_loss . By default, huberslossfunction_border=10 but this is only an arbitrary default value that should be changed.
	// The errorfunction is then given by Huber's loss function/pointnumber,
	// where pointnumber is the number of points fitted.
	// 
	// logcosh is a loss function that sets log(cosh(y[i]-f(X[i]))/pointnumber as size of the error at point (X[i],Y[i]), where f is the function that is fitted
	// and pointnumber is the number of points used in the fit.
	// 
	// if the quantile loss function is used, we need the gamma parameter and set the loss per point 
	// to (gamma-1)* (y[i]-f(X[i])/pointnumber if y[i]-f(X[i]<0 and to gamma*(y[i]-f(X[i])/pointnumber otherwise. f is the function that is fitted and pointnumber
	// is the number of points used in the fit. Note that quantile is an asymmetric loss function. Therefore, it should mostly be used with the linear
	// robust curve fitting algorithms, since then it is only used for outlier removal. If the quantile loss function is used with the non-linear robust algorithms
	// it is likely to confuse the Levenberg-Marquardt algorithm.
	// 
	// if custom is used, the function pointer loss_pp is called as loss_pp(Y[i],f(X[i],pointnumber) which should compute an user defined 
	// loss function of the datapoint (X[i],Y[i]), where f(X[i]) is the predicted values from the fit at X[i], pointnumber is the number of points that are
	// fitted. This is information is included so that the loss function can scale with the pointnumber, if needed.
	// 
	// if custom is used and agg_err is set to null, the entire error between the predicted values f(X) and the data Y for all indices
	// X[i] will be set simply to the sum of the losses per point. 
	// if custom is used and  agg_err is not set to Null, then it is called with a reference to a double valarray that contains the losses per point 
	// as given by calls of loss_pp for all data points. The function agg_err should then compute the value of the total error in return.
	// 
	// Note that if robst regression methods are used, it is best if the aggregate error is some kind of average error. The robust curve fitting methods
	// can remove points based on this quantity. If, e.g the lossperpoint is the square of the errors per point, and agg_err is just their sum, without
	// a division by the number of points, then, if all points do not provide a 100% exact fit, one would reduce the error usually in any case if 
	// one just removes the number of points. This would, however, not increase the precision of the fit, but it would usually imply that not enough points
	// would be used. So it is best to set the aggregate error agg_err to some kind of average, which can be achived e.g. by summing over the given residuals 
	// and then dividing by the pointnumber, i.e. by the size of the given array. or by scaling the value returned of lossperpoint by the pointnumber.
	enum errorfunction_name
	{
		huberlossfunction,
		squaredresidual,
		absolutevalue,
		logcosh,
		quantile,
		custom
	};


	// custom callback function that returns the loss  for a single point given the curve fit. Y is the Y value of a point, Y_pred the prediction, 
	// from the curve fit and pointnumber the number of all points that are fitted
	typedef function<double( const double Y,  double Y_pred, size_t pointnumber)> lossperpoint;



	// callback function that returns the entire error from an array of values storing the results of calls to lossperpoint
	typedef function<double(valarray<double>& errrorarray)> aggregate_error;



	// The struct errorfunction contains the specifics of the loss function.
	// 
	// The field lossfunction: describes the name of the loss function. Note that if the lossfunction is given an argument of linear robust 
	// regression algorithms, then the lossfunction is only used for the outlier detection, since the linear curve fitting algorithms that form the basis
	// of the robust methods are either simple linear regression or Siegel's repeated median algorithm, that have their
	// own error metric which they minimize.
	// In contrast, when a non-linear robust algorithm is used, the loss function is also used for the Levenberg-Marquardt algorithm.
	//
	// The field huberslossfunction_border specifies the delta parameter of Huber's loss function if it is set as loss funciton
	// The field gamma specifies the gamma parameter if quantile is set as a loss function,
	// The fields loss_perpoint is a function pointer to a custom loss function given a point. It is only used when custom is used as a loss function
	// The field aggregate_err is a function pointer to a function that computes the entire error given the array of residuals computed by loss_perpoint.
	// it is only used when custom is used. If custom is set as a loss function and aggregate_err is still Null, then the library will just sum the results
	// of the calls of loss_perpoint and use this sum as entire error. 
	// Note that if robust regression methods are used, it is best if the aggregate error is some kind of average error. The robust curve fitting methods
	// can remove points based on this quantity. If, e.g the lossperpoint is the square of the errors per point, and agg_err is just their sum, without
	// a division by the number of points, then, if all points do not provide a 100% exact fit, one would reduce the error usually in any case if 
	// one just removes the number of points. This would, however, not increase the precision of the fit, but it would usually imply that not enough points
	// would be used. So it is best to set the aggregate error agg_err to some kind of average, which can be achived e.g. by summing over the given residuals 
	// and then dividing by the pointnumber, i.e. by the size of the given array. or by scaling the value returned of lossperpoint by the pointnumber.
	struct errorfunction
	{
		errorfunction_name lossfunction{ squaredresidual };
		double huberslossfunction_border{ 10.0 };
		double gamma{ 0.25 };
		lossperpoint loss_perpoint{NULL};
		aggregate_error aggregate_err{NULL};
	};

};
