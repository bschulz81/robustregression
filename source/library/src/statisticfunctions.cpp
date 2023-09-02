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
#include <valarray>
#include <omp.h>
#include <stdint.h>
#include <float.h>
#include <execution>
#include "statisticfunctions.h"

#ifdef GNUCOMPILER
#define _inline inline
#endif

#ifdef CLANGCOMPILER
#define _inline inline
#endif


using namespace std;
using namespace Statisticfunctions;

double G(double y, size_t nu);

double H(double y, size_t nu);

ROBUSTREGRESSION_API inline  double Statisticfunctions::fabs(double f) {
	return (f < 0) ? -f : f;
}

inline  double  Statisticfunctions::factorial(size_t n)
{
	double ret = 1.00;
	for (size_t i = 2; i <= n; ++i)
		ret *= (double)i;
	return ret;
}


ROBUSTREGRESSION_API inline  double  Statisticfunctions::stdeviation(valarray<double>& errs)
{
	size_t s = errs.size();
	double sum = 0, devi = 0, t = 0;
	for (size_t i = 0; i < s; i++)
	{
		sum += (errs)[i];
	}
	double average = sum / (double)s;

	for (size_t i = 0; i < s; i++)
	{
		t = ((errs)[i] - average);
		t *= t;
		devi += t;
	}
	return sqrt(devi / s);

}


ROBUSTREGRESSION_API inline  double  Statisticfunctions::average(valarray<double>& errs)
{
	size_t s = errs.size();
	double sum = 0, devi = 0, t = 0;
	for (size_t i = 0; i < s; i++)
	{
		sum += (errs)[i];
	}
	return sum / (double)s;
}


ROBUSTREGRESSION_API inline  double  Statisticfunctions::median(valarray<double> arr)
{
	size_t n = arr.size();
	if (n == 1)
		return arr[0];
	else
	{
		size_t nhalf = n / 2;

		if (n % 2 != 0)
		{
#if (__cplusplus == 201703L) && !defined(MACOSX)
			nth_element(std::execution::par, std::begin(arr), std::begin(arr) + nhalf, std::begin(arr) + n);
#else
			nth_element(std::begin(arr), std::begin(arr) + nhalf, std::begin(arr) + n);
#endif
			return  (arr)[nhalf];
		}
		else
		{
			std::sort(std::begin(arr), std::end(arr));
			return (arr[(size_t)((n - 1) / 2)] + arr[(size_t)(nhalf)]) / 2;
		}
	}

}



inline double  Statisticfunctions::lowmedian(valarray<double> arr)
{
	size_t n = arr.size();
	size_t m = (size_t)(floor(((double)n + 1.0) / 2.0) - 1.0);

#if __cplusplus == 201703L && !defined(MACOSX)
	nth_element(std::execution::par, std::begin(arr), std::begin(arr) + m, std::begin(arr) + n);
#else
	nth_element(std::begin(arr), std::begin(arr) + m, std::begin(arr) + n);
#endif
	return (double)(arr)[m];
}


inline double cot(double x)
{
	return  cos(x) / sin(x);
}

ROBUSTREGRESSION_API inline  double  Statisticfunctions::t(double alpha, size_t nu)
{
	const double PI = 3.14159265358979323846;
	if (nu % 2 != 0)
	{
		double zeta = 0;
		double zeta1 = cot(alpha * PI);
		do
		{
			zeta = zeta1;
			zeta1 = sqrt((double)nu) * cot(alpha * PI + H(zeta, nu));
		} while (Statisticfunctions::fabs(zeta1 - zeta) > 0.0001);
		return zeta1;
	}
	else
	{
		double zeta = 0;
		double zeta1 = sqrt(2.0 * pow(1.0 - 2.0 * alpha, 2.0) / (1 - pow(1.0 - 2.0 * alpha, 2.0)));
		do
		{
			zeta = zeta1;
			zeta1 = pow(1.0 / nu * (pow(G(zeta, nu) / (1.0 - 2.0 * alpha), 2.0) - 1.0), -0.5);
		} while (Statisticfunctions::fabs(zeta1 - zeta) > 0.0001);
		return zeta1;
	}
}

ROBUSTREGRESSION_API inline  double   Statisticfunctions::crit(double alpha, size_t N)
{
	double temp = pow(t(alpha / (2.0 * (double)N), N - 2), 2.0);
	return ((double)N - 1.0) / sqrt((double)N) * sqrt(temp / ((double)N - 2.0 + temp));
}


ROBUSTREGRESSION_API inline double  Statisticfunctions::peirce(size_t pointnumber, size_t numberofoutliers, size_t fittingparameters)
{
	double diff1 = (double)(pointnumber - numberofoutliers);

	double res = 0.0;

	double Q = pow(numberofoutliers, (numberofoutliers / pointnumber)) * pow(diff1, (diff1 / pointnumber)) / pointnumber;
	double a = 1.0, b = 0.0;
	while (Statisticfunctions::fabs(a - b) > (pointnumber * 2.0e-15))
	{
		double l = pow(a, numberofoutliers);
		if (l == 0.0)
		{
			l = 1.0e-5;
		}
		res = 1.0 + (diff1 - fittingparameters) / numberofoutliers * (1.0 - pow(pow((pow(Q, pointnumber) / l), (1.0 / diff1)), 2.0));
		if (res >= 0)
		{
			b = a;
			a = exp((res - 1.0) / 2.0) * erfc(sqrt(res) / sqrt(2.0));
		}
		else
		{
			b = a;
			res = 0.0;
		}
	}
	return res;
}



ROBUSTREGRESSION_API inline  size_t  Statisticfunctions::binomial(size_t n, size_t k)
{
	if (k == 0)
	{
		return 1;
	}
	else if (n == k)
	{
		return 1;
	}
	else
	{
		if (n - k < k)
		{
			k = n - k;
		}
		double prod1 = 1.0, prod2 = (double)n;
		for (size_t i = 2; i <= k; i++)
		{
			prod1 *= i;
			prod2 *= ((double)n + 1.0 - (double)i);
		}
		return(size_t)(prod2 / prod1);
	}
}


ROBUSTREGRESSION_API inline  double  Statisticfunctions::Q_estimator(valarray<double>& err)
{
	size_t s = err.size();
	valarray<double> t1(binomial(s, 2));





	size_t i = 0;
	for (size_t w = 0; w < s - 1; w++)
	{
		for (size_t k = w + 1; k < s; k++)
		{
			t1[i] = (Statisticfunctions::fabs((err)[w] - (err)[k]));
			i++;
		}
	}
	size_t h = s / 2 + 1;
	size_t k = binomial(h, 2) - 1;

#if __cplusplus == 201703L && !defined(MACOSX)
	std::nth_element(std::execution::par, std::begin(t1), std::begin(t1) + k, std::end(t1));
#else
	std::nth_element(std::begin(t1), std::begin(t1) + k, std::end(t1));
#endif


	double cn[] = { 0,0,0.399, 0.994, 0.512 ,0.844 ,0.611, 0.857, 0.669 ,0.872 };
	double cc = 1;
	if (s <= 9)
	{
		cc = cn[s];
	}
	else
	{
		if (s % 2 != 0)
		{
			cc = s / (s + 1.4);
		}
		else
		{
			cc = s / (s + 3.8);
		}
	}
	return cc * 2.2219 * t1[k];
}



ROBUSTREGRESSION_API inline   double    Statisticfunctions::S_estimator(valarray<double>& err)
{
    valarray<double> x(err);
    size_t n = x.size();
    std::valarray<double> a2(n); 

    std::sort(std::begin(x), std::begin(x) + n);

	a2[0] = x[n / 2] - x[0];
	a2[n - 1] = x[n - 1] - x[(n + 1) / 2 - 1];
    


	for (size_t i = 1; i < (n + 1) / 2; i++)
	{
		size_t leftB, rightB,
			nB = n - i - 1,
			diff2 = (nB - i) / 2,
			Amin = diff2,
			Amax = diff2 + i,
			leftA = leftB = 1,
			rightA = rightB = nB;
		while (leftA < rightA)
		{
			size_t l = rightA - leftA,
				half = l / 2,
				even = l % 2,
				tryA = leftA + half,
				tryB = leftB + half;
			if (tryA > Amin)
			{
				if (tryA > Amax)
				{
					rightA = tryA;
					leftB = tryB + even;
				}
				else
				{
					if (x[i] - x[i - tryA + Amin] < x[tryB + i] - x[i])
					{
						rightB = tryB;
						leftA = tryA + even;
					}
					else
					{
						rightA = tryA;
						leftB = tryB + even;
					}
				}
			}
			else
			{
				rightB = tryB;
				leftA = tryA + even;
			}
		}
		if (leftA > Amax)
			a2[i] = x[leftB + i] - x[i];
		else
			a2[i] = std::min(x[i] - x[i - leftA + Amin], x[leftB + i] - x[i]);
	}

	for (size_t i = (n + 1) / 2; i < n - 1; i++)
	{
		size_t leftB,rightB, nA = n - i - 1,
			diff2 = (i - nA) / 2,
			leftA = leftB=1,
			rightA = rightB= i,
			Amin = diff2 + 1,
			Amax = diff2 + nA;
		while (leftA < rightA)
		{
			size_t l = rightA - leftA,
				half = l / 2,
				even = l % 2,
				tryA = leftA + half,
				tryB = leftB + half;
			if (tryA < Amin)
			{
				rightB = tryB;
				leftA = tryA + even;
			}
			else
			{
				if (tryA > Amax)
				{
					rightA = tryA;
					leftB = tryB + even;
				}
				else
				{
					if (x[i + tryA - Amin + 1] - x[i] < x[i] - x[i - tryB])
					{
						rightB = tryB;
						leftA = tryA + even;
					}
					else
					{
						rightA = tryA;
						leftB = tryB + even;
					}
				}
			}
		}
		if (leftA > Amax)
			a2[i] = x[i] - x[i - leftB];
		else
			a2[i] = std::min(x[i + leftA - Amin + 1] - x[i], x[i] - x[i - leftB]);
	}

	double cn = 1.0;
     if (n <= 9) 
    {
        double mynum[10] = {0,0, 0.743, 1.851, 0.954,1.351, 0.993, 1.198 ,1.005, 1.131 };
        cn = mynum[n];
    }
    else {
        if (n % 2 == 1) 
            cn = n / (n - 0.9);
    }

   return cn * 1.1926 * lowmedian(a2);
}


ROBUSTREGRESSION_API inline   double    Statisticfunctions::MAD_estimator(valarray<double>& err, double& m)
{
	size_t s = err.size();
	valarray<double>m1(s);
	for (size_t i = 0; i < s; i++)
	{
		m1[i] = (Statisticfunctions::fabs((err)[i] - m));
	}
	double cn[] = { 0, 0, 1.196 ,1.495 ,1.363, 1.206, 1.200, 1.140, 1.129,1.107 };
	double c = 1.0;
	if (s <= 9)
	{
		c = cn[s];
	}
	else
	{
		c = s / (s - 0.8);
	}

	return  c * 1.4826 * median(m1);
}



ROBUSTREGRESSION_API inline double  Statisticfunctions::Q1(valarray<double> m)
{
	size_t s = m.size();
#if (__cplusplus == 201703L) && !defined(MACOSX)
	std::sort(std::execution::par,std::begin(m), std::end(m));
#else
	std::sort(std::begin(m), std::end(m));
#endif
	double Q1;
	if (s < 4)
	{
		if (s == 3)
		{
			Q1 = m[0];
		}
		else if (s == 2)
		{
			Q1 = m[0];

		}
		else
		{
			Q1 = m[0];
		}
	}
	else
	{
		float test1 = s * 0.25;
		if (test1 == (size_t)(test1))
			Q1 = (m[(size_t)test1 - 1] + m[(size_t)test1]) / 2.0;
		else
			Q1 = m[((size_t)(test1))];
	}
	return (Q1);
}

ROBUSTREGRESSION_API inline double  Statisticfunctions::Q3(valarray<double> m)
{
	size_t s = m.size();
#if (__cplusplus == 201703L) && !defined(MACOSX)
	std::sort(std::execution::par,std::begin(m), std::end(m));
#else
	std::sort(std::begin(m), std::end(m));
#endif
	double Q3;
	if (s < 4)
	{
		if (s == 3)
		{
			Q3 = m[2];
		}
		else if (s == 2)
		{
			Q3 = m[1];
		}
		else
		{
			Q3 = m[0];
		}
	}
	else
	{

		float test2 = s * 0.75;
		if (test2 == (size_t)(test2))
			Q3 = (m[(size_t)test2 - 1] + m[(size_t)test2]) / 2.0;
		else
			Q3 = m[((size_t)(test2))];
	}
	return Q3;
}



ROBUSTREGRESSION_API inline  double   Statisticfunctions::onestepbiweightmidvariance(valarray<double>& err, double& m)
{
	size_t s = err.size();

	double mad = MAD_estimator(err, m);
	double p1 = 0.0, p2 = 0.0;
	for (size_t i = 0; i < s; i++)
	{
		double ui = ((err)[i] - m) / (9.0 * mad);
		if (Statisticfunctions::fabs(ui) < 1.0)
		{
			double ui2 = ui * ui;
			double g1 = (err)[i] - m;
			double g2 = (1.0 - ui2);
			p1 += g1 * g1 * pow(g2, 4.0);
			p2 += g2 * (1.0 - 5.0 * ui2);
		}
	}

	return (sqrt(s) * sqrt(p1) / Statisticfunctions::fabs(p2));

}


ROBUSTREGRESSION_API inline  double  Statisticfunctions::T_estimator(valarray<double>& err)
{
	size_t s = err.size();
	valarray<double>med1(s);

	valarray<double>arr(s - 1);

	for (size_t i = 0; i < s; i++)
	{
		size_t q = 0;

		for (size_t j = 0; j < s; j++)
		{
			if (i != j)
			{
				arr[q] = (fabs((err)[i] - (err)[j]));
				q++;
			}
		}
		med1[i] = (median(arr));
	}

	size_t h = s / 2 + 1;
	double w = 0;
#if __cplusplus == 201703L && !defined(MACOSX)
	sort(std::execution::par, std::begin(med1), std::end(med1));
#else
	sort(std::begin(med1), std::end(med1));
#endif

	for (size_t i = 1; i <= h; i++)
	{
		w += med1[i - 1];
	}

	return  (1.38 / ((double)h)) * w;
}


inline double  G(double y, size_t nu)
{
	double sum = 0;
	for (size_t j = 0; j <= nu / 2 - 1; j++)
	{
		sum += factorial((size_t)2.0 * j) / (pow(2.0, 2.0 * j) * pow(factorial(j), 2)) * pow(1 + y * y / nu, -((double)j));
	}
	return sum;
}



inline double  H(double y, size_t nu)
{
	double sum = 0;

	for (size_t j = 1; j <= size_t((nu + 1) / 2 - 1); j++)
	{
		sum += factorial((size_t)j) * factorial(size_t(j - 1.0)) / (pow(4, -((double)j)) * factorial((size_t)2.0 * j)) * pow((1 + y * y / nu), -((double)j));
	}
	return y / (2 * sqrt((double)nu)) * sum;
}
