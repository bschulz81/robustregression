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
#include <valarray>
#include <omp.h>

#include "lossfunctions.h"
#include "linearregression.h"
#include "statisticfunctions.h"
#ifdef GNUCOMPILER
#define _inline inline
#endif

#ifdef CLANGCOMPILER
#define _inline inline
#endif


using namespace std;

ROBUSTREGRESSION_API inline  bool Linear_Regression::linear_regression(const valarray<double>& x, const valarray<double>&y, Linear_Regression::result&res)
{
	
	size_t usedpoints = x.size();



	double sumy = 0;
	for (size_t n = 0; n < usedpoints; n++)
	{
		sumy += y[n];
	}
	double yaverage = sumy / (double)usedpoints;



	double sumx = 0, sumxy = 0, sumxx = 0;
	for (size_t n = 0; n < usedpoints; n++)
	{
		sumx += x[n];
		sumxy += x[n] * y[n];
		sumxx += x[n] * x[n];
	}

	double xaverage = sumx / (double)usedpoints;

	double t = sumxx - sumx * xaverage;


	double thisslope = (sumxy - sumx * yaverage) / t;
	double thisintercept = yaverage - thisslope * xaverage;

	res.main_intercept = thisintercept;
	res.main_slope = thisslope;
	return true;

}


ROBUSTREGRESSION_API inline  bool Linear_Regression::median_linear_regression(const valarray<double>&x,const valarray<double>&y, Linear_Regression::result&res)

{
	size_t usedpoints = x.size();
	valarray<double> stacks2(usedpoints);
	valarray<double> stacks1(usedpoints - 1);

	size_t halfsize = usedpoints / 2;

	valarray<double> stacki1(usedpoints);
	
	for (size_t i = 0; i < usedpoints; i++)
	{
		size_t q = 0;
		for (size_t j = 0; j < usedpoints; j++)
		{
			double t = x[j] - x[i];
			if (t != 0)
			{
				stacks1[q] = (y[j] - y[i]) / t;
				q++;
			}
		}
		stacks2[i] = Statisticfunctions::median(stacks1, q);
	}

	double thisslope = Statisticfunctions::median(stacks2, usedpoints);
	for (size_t n = 0; n < usedpoints; n++)
	{
		stacki1[n] =y[n] - thisslope * x[n];
	}
	double thisintercept = Statisticfunctions::median(stacki1, usedpoints);
	res.main_intercept = thisintercept;
	res.main_slope = thisslope;
	return true;

}
