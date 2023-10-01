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
#include <string>

#include "lossfunctions.h"
#include "robustregressionlib_exports.h"

using namespace std;

namespace Linear_Regression
{
	//stores the result of the linear regression of a function f(x)=main_slope*x+main_intercept
	struct result
	{
		double main_slope{ 0 };
		double main_intercept{ 0 };

	};
	


	// computes a simple linear regression. datapoints are given by two valarrays x and y. the result is put into res.
	ROBUSTREGRESSION_API bool linear_regression(const valarray<double>& x, const valarray<double>& y, Linear_Regression::result& res);

	// computes Siegel's repeated median linear regression, which is more robust against outliers.
	// Datapoints are given by two valarrays x and y. the result is put into res. For a documentation of the algorithm, see
	// Andrew F.Siegel.Robust regression using repeated medians.Bionaetrika, 69(1) : 242–244, 1982,
	// Andrew Stein and Michael Werman. 1992. Finding the repeated median regression line.In 
	// Proceedings of the third annual ACM - SIAM symposium on Discrete algorithms(SODA '92). Society for Industrial and Applied Mathematics, USA, 409–413.
	// https://dl.acm.org/doi/10.5555/139404.139485 Note that a comparably slow algorithm is used in the current implementation of the library.
	ROBUSTREGRESSION_API  bool median_linear_regression(const valarray<double>&x,const  valarray<double>&y, Linear_Regression::result&res);
}
