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

	//the name of the metric of how the main error is computed.
	enum errorfunction_name
	{
		huberlossfunction,
		squaredresidual,
		absolutevalue,
	};



	//the specifics of the errorfunction, contains the name of the metric, and the border delta of huber's loss function if it is used.
	struct errorfunction
	{
		errorfunction_name lossfunction{ squaredresidual };
		double huberslossfunction_border{ 10.0 };
	};

};
