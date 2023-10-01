# RobustregressionLib
This is a c++ library with statistical machine learning algorithms for linear and non-linear robust regression.

It implements the statistical algorithms that were originally developed by the author for an autofocus application for telescopes

and published in 	arXiv:2201.12466 [astro-ph.IM], https://doi.org/10.1093/mnras/stac189

In addition to these, two other robust algorithms were added and the curve fitting library has been brought into a form of a
clear and simply API that can be easily used for very broad and general applications.

The library offers Python bindings for most functions. So the programmer has the choice between c++ and Python. In order to 
compile the library with Python bindings Pybind11 and Python3 should be installed and be found by CMake. 
Otherwise, only the C++ standard template library is used, together with OpenMP. 

The documentation of the functions that the library uses are written in the C++header files and in the __doc__ methods of the Python bindings.

In addition, a c++ application and a Python script is provided that show the functions of the library with very simple data.

The Library is released under MIT License.

Apart from his own publication, the author has not found the main robust curve fitting algorithms from this library in the statistical literature.

One of the algorithms presented here is a modification of the forward search algorithms by  Hadi and Simonoff, Atkinson and Riani and the least trimmed squares
method of Rousseeuw. The modification of the author is to use various estimators to include data after the algorithm tried a random initial combination.

The algorithm was originally developed for physical problems, where one has outliers but also data, which is often subject to random fluctuations, like astronomical seeing.
As we observed during trials with the astronomy application, including the S estimator in the forward search removes large outliers but allows for small random fluctuations 
in the data, which resulted in more natural curve fits than if we would simply select the "best" model that we would get from the forward search. If some degree of randomness is present,
the "best" model chosen by such a method would have the smallest error almost certainly by accident and would not include enough points for a precise curve fit.
The usage of the statistical estimators in the forward search appeared to prevent this problem.

The modified least trimmed squares method has also been used by the author in arXiv:2201.12466 with the various estimators to judge the quality of measurement data, which was 
defined as "Better" when the algorithm, if used sucessively with several different estimators, comes to a more similar result. 

Another algorithm presented in this library is an iterative method which also employs various estimators. It has the advantage that it should work with larger datasets but its statistical 
properties have not been extensively tested yet.

Because of the use of various statistical estimators and methods, the library builds on previous articles from the statistical literature. 
Some references are:

1. Smiley W. Cheng, James C. Fu, Statistics & Probability Letters 1 (1983), 223-227, for the t distribution
2. B. Peirce,  Astronomical Journal II 45 (1852) for the peirce criterion
3. Peter J. Rousseeuw, Christophe Croux, J. of the Amer. Statistical Assoc. (Theory and Methods), 88 (1993), p. 1273, for the S, Q, and T estimator
5. T. C. Beers,K. Flynn and K. Gebhardt,  Astron. J. 100 (1),32 (1990), for the Biweight Midvariance
6. Transtrum, Mark K, Sethna, James P (2012). "Improvements to the Levenberg-Marquardt algorithm for nonlinear least-squares minimization". arXiv:1201.5885, for the Levenberg Marquardt Algorithm,
7. Rousseeuw, P. J. (1984).Journal of the American Statistical Association. 79 (388): 871–880. doi:10.1080/01621459.1984.10477105. JSTOR 2288718.
   Rousseeuw, P. J.; Leroy, A. M. (2005) [1987]. Robust Regression and Outlier Detection. Wiley. doi:10.1002/0471725382. ISBN 978-0-471-85233-9, for the least trimmed squares algorithm
8. Hadi and Simonoff, J. Amer. Statist. Assoc. 88 (1993) 1264-1272, Atkinson and Riani,Robust Diagnostic Regression Analysis (2000), Springer, for the forward search
9. Croux, C., Rousseeuw, P.J. (1992). Time-Efficient Algorithms for Two Highly Robust Estimators of Scale. In: Dodge, Y., Whittaker, J. (eds) Computational Statistics. Physica, Heidelberg. https://doi.org/10.1007/978-3-662-26811-7_58 
 (For the faster version of the S and Q estimators.) The versions of the S and Q estimators in this library are now adapted from Croux and Rousseeuw to the C language. Note that it is not the same Code because of some optimizations. Since many variables act on array indices in this algorithm, it was actually non-trivial to convert from Fortran to C. Especially for the Q estimator, the naive algorithm is faster for less than 100 datapoints. For the S estimator this is the case for less than 10 datapoints. Therefore, in these cases the naive versions are still used.
10. Andrew F. Siegel. Robust regression using repeated medians. Bionaetrika, 69(1):242–244, 1982,Andrew Stein and Michael Werman. 1992. Finding the repeated median regression line. In Proceedings of the third annual ACM-SIAM symposium on Discrete algorithms (SODA '92). Society for Industrial and Applied Mathematics, USA, 409–413. https://dl.acm.org/doi/10.5555/139404.139485

# Compiling and Installing the library:

The Library needs CMake and a C compiler that is at least able to generate code according to the C14 standard
(per default, if one does not use Clang or MacOs, it uses C17, but with a CXX_STANDARD 14 switch set in the 
CMakeLists.txt for the library, it can use C14, which is the the default for Clang and MacOS.) 

The library also makes use of OpenMP and needs Python in version 3 and pybind11 to compile. 

By default, the library also containts two test applications. 

If the variable $WITH_TESTAPP, a c++ test application is compiled and put in an /output directory. 

The library also shipps with a Python module. By default, the CMake variable $WithPython is ON and a Python module will
be generated in addition to a c++ library.

If $WITH_TESTAPP and $WithPython are set, which is the default, then a Python test application will be generated in addition to
the C++ test application.

## Installing with CMake
One can compile the library also traditionally via CMake. Typing 

> cmake . 

in the package directory will generate the files necessary to compile the library, depending on the CMake generator set by the user.

Under Linux, the command

> make .

will then compile the library.

After compilation, an /output directory will appear with the library in binary form. 
By default, the library also containts two test applications. 

If the variable $WITH_TESTAPP is set, a c++ test application is compiled and put in an /output directory. 

The library also ships with a Python module. By default, the CMake variable $With_Python is ON and a Python module will
be generated in addition to a c++ library.

If $WITH_TESTAPP and $With_Python are set to ON, which is the default, then a Python test application will be generated and compiled into the /output directory.

By compiling with CMake, the Python module is, just compiled into the /output directory. It is not installed in a path for system libraries
or python packages. So if one wants to use the Python module, one has either a) to write the script in the same folder where the module is, or b) load
it by pointing Python to the explicit path of the module, or c) copy the module to a place where Python can find it.


If one does not want the Python module to be compiled, one may set the cmake variable With_Python to OFF.

## Installing with PIP (This option is mostly for Windows since Linux distributions have their own package managers)

If one wants that the module is installed into a library path, where Python scripts may be able to find it easily, one can compile and install the module also 
with pip instead of CMake by typing

> pip install .

in the package directory. 

After that, the module is compiled and the binaries are copied into a path for Python site-packages, where Python scripts should be able to find it.

This is successfull on Windows.

Unfortunately, problems to find the module remain on Linux.
If pip is called by the  root user, the module is copied into the /usr/lib directory. Despite this,  python scripts have difficulties to load the module. 
If one does not install the module as root user, pip will install it in a local site-package directory, where python also has problems to find the module.

If the module was compiled by pip, one can uninstall it by typing

> pip uninstall pyRobustRegressionLib

Under Linux, compiling with cmake should be preferred. Not at least because linux package managers (e.g. emerge) sometimes have conflicts with pip.

Additionally, the python environment will select ninja as a default generator, which will require to clean the build files
if an earlier generation based on cmake was done that may have used a different generator.


# Documentation of the library functions

## For the Python language

### Calling the documentation
Documentation of the API is provided in C++ header files in the /library/include directory and the docstrings for the Python module in the src/pyRobustRegressionLib
Module. The latter It can be loaded in Python scripts with 

> import pyRobustRegressionLib as rrl

The command 

> print(rrl.\_\_doc__)

Will list the sub-modules of the library, which are 

- StatisticFunctions, 
- LinearRegression, 
- MatrixCode, 
- NonLinearRegression and 
- RobustRegression
- LossFunctions

And their docstrings can be called e.g. by
> print(rrl.*SubModuleName*.\_\_doc__)

e.g.

> print(rrl.StatisticFunctions.\_\_doc__).

Will list the functions and classes of the sub-module StatisticFunctions. The free functions and classes all have more detailed doc
strings that can be called as below for example

> print (rrl.MatrixCode.Identity.\_\_doc__)

More convenient documentation is provided in the header files of the C++ source code of the package,
which can be found in the /library/include directory.

The header files can be found in the include subdirectory of the package.

In the testapp folder, two example programs, one in Python and one in C++ is provided.
These test applications have extensive comments and call many functions of the librarym which show the basic usage. 

The curve fits that are done in the provided example programs are, however, very simple of course.
This was done in order to keep the demonstration short and simple.
The library is of course intended to be used with larger and much more complicated data.

### Simple linear regression

Let us now define some vector for data X and > which we want to fit to a line.

> print("\nDefine some arrays X and Y")
> 
> X=[-3.0,5.0,7.0,10.0,13.0,16.0,20.0,22.0]
> 
> Y=[-210.0,430.0,590.0,830.0,1070.0,1310.0,1630.0,1790.0]

A simple linear fit can be called as follows:
At first we create an instance of the result structure, where the result is stored.

> res=rrl.LinearRegression.result()

Then, we call the linear regression function
> rrl.LinearRegression.linear_regression(X, Y, res)



And finally, we print out the slope and intercept
> print("Slope")
> 
> print(res.main_slope)
> 
> print("Intercept")
> 
> print(res.main_intercept)

### Robust regression
The robust regression is just slightly more complicated. Let us first add two outliers into the dats:

> X2=[-3.0, 5.0,7.0, 10.0,13.0,15.0,16.0,20.0,22.0,25.0]
> 
> Y2=[ -210.0, 430.0, 590.0,830.0,1070.0,20.0,1310.0,1630.0,1790.0,-3.0]


#### Median regression
For linear regression, the library also has a median linear regression function, which can be called in the same way

> rrl.LinearRegression.median_linear_regression(X2, Y2, res)

but is slightly more robust.


#### Modified forward search/modified Lts regression
Median linear regression is a bit slower as simple linear regression and can get wrong if many outliers are present.

Therefore, the library has two methods for robust regression that can remove outliers.

The  structure that stores the result for robust linear regression now includes the indices of the used and rejected point.

We instantiate it with

> res= rrl.RobustRegression.linear_algorithm_result()

Additionally, there is a struct that determines the behavior of the algorithm.  
Upon construction without arguments, it gets filled with default values.

> ctrl= rrl.RobustRegression.modified_lts_control_linear()

By default, the S-estimator is used with an outlier_tolerance parameter of 3, and the method can find 30% of the points as outliers at maximum. 
But all this can be changed, of course

Now we call the modified forward search/ modified lts algorithm
> rrl.RobustRegression.modified_lts_regression_linear(X2, Y2, ctrl, res)

and then print the result, first the slope and intercept 
> print("Slope")
> 
> print(res.main_slope)
> 
> print("Intercept")
> 
> print(res.main_intercept)

Then the indices of the outliers

> print("\nOutlier indices")
> 
> for ind in res.indices_of_removedpoints:
> 
> &emsp;print(ind)

When we want to change the estimators, or the outlier tolerance parameter, the loss function, or the maximum number of outliers we can find, or other details, we can simply
set this in the control struct.

By default, the S estimator is used with an outlier_tolerance of 3 in the same formula i.e. one has

> ctrl.rejection_method=rrl.RobustRegression.estimator_name.tolerance_is_decision_in_S_ESTIMATION

and a point is an outlier if 
> |err-median(errs)|/S_estimator(errs)>outlier_tolerance

where err is the residuum of the point and errs is the array of residuals. 
They are measured by a specified loss function. If none was given, squared errors are used by default. 

By default, the outlier_tolerance parameter is set to 3.

If we want to have a different value, e.g. 3.5, for the outlier_tolerance parameter, we can easily set e.g.

> ctrl.outlier_tolerance=3.5

The command below would imply that the Q estimator is used:

> ctrl= rrl.RobustRegression.modified_lts_control_linear()
> 
> ctrl.rejection_method=rrl.RobustRegression.estimator_name.tolerance_is_decision_in_Q_ESTIMATION

Then a point is an outlier if, for its residual with the curve fit err, we have,given the array of all residuals errs:
> |err-median(errs)|/Q_estimator(errs)>outlier_tolerance


The command below would change the estimator to the interquartile range method when the modified lts/modified forward search algorithm is used. 

With the setting below, a point is an outlier if its residual is below Q1 − tolerance* IQR or above Q3 + tolerance IQR.
If we do not change the loss function, least squares is used by default.

> ctrl= rrl.RobustRegression.modified_lts_control_linear()
> 
> ctrl.rejection_method=rrl.RobustRegression.estimator_name.tolerance_is_interquartile_range

 For the interquartile range estimator, we should set the tolerance usually to 1.5

> ctrl.outlier_tolerance=1.5

before we call the regression function.

Similarly, the loss function can be changed. For example, the absolute value of the residuals is usually more statistically robust than the square of the residuals

> ctrl.lossfunction=rrl.LossFunctions.absolutevalue

changes the lossfunction to the absolute value. 

One can also specify Huber's loss function, but then one also has to supply a border parameter beyond which the
function starts its linear behavior.
One also can set a log cosh loss function, or a quantile loss function. The latter needs a gamma parameter to be specified within the interval of 0 and 1.
Finally, one can define a custom loss function with a callback mechanism.

Note that if we use the linear versions of the robust regression, then these methods would just make simple linear
fits or repeated median fits, which minimize their own loss function, and the selected loss function of the library
is then only used for outlier removal.

With the robust non-linear regression algorithms, the custom error functions are used for the curve fits as well 
as for the outlier removal procedures.

If one needs a linear fit where the custom error function is used as a metric for the curve fit as well as for the outlier
removal, one has to use the non-linear algorithm with a linear call back function. 

Note also that the quantile loss function is asymmetric. Therefore, the quantile loss function should mostly be used with
the linear robust curve fitting algorithms, since then it is only used for outlier removal. 
If the quantile loss function is used with the non-linear robust algorithms it is likely to confuse the Levenberg-Marquardt algorithm 
because of its asymmetry.




Note that the forward search can be very time consuming, as its performance is given by the binomial koefficient of the pointnumber over the maximum number of 
outliers it can find (which per default is 30% of the pointnumber).

#### Iterative outlier removal

A faster algorithm is the iterative outlier removal method, which makes a curve fit with the entire points, then removes the points whose residuals are outliers and
then makes another curve fit with the remaining point set until no outliers are found anymore.

It can be called similarly:

We define the control structure. Note that it is different from the modified forward search/modified lts control structure. 
> ctrl= rrl.RobustRegression.linear_algorithm_control()

And again create a structure for the result
> res= rrl.RobustRegression.linear_algorithm_result()
Then we start the algorithm
> rrl.RobustRegression.iterative_outlier_removal_regression_linear(X2, Y2, ctrl, res)

And print the result
> print("Slope")
> 
> print(res.main_slope)
> 
> print("Intercept")
> 
> print(res.main_intercept)
> 
> print("\nOutlier indices")
> 
> for ind in res.indices_of_removedpoints:
> 
> &emsp;print(ind)



### Non Linear Regression
Non-linear regression uses an implementation of the Levenberg-Marquardt algorithm

The Levenberg-Marquardt algorithm needs an initial guess for an array of parameters beta , a function f(X,beta) to be fitted and a Jacobi matrix J(X,beta)
For example, a curve fit can be made by  initialising the result, control and initdata structures as follows:
After a call of the constructors:

> res=rrl.NonLinearRegression.result()
> 
> ctrl=rrl.NonLinearRegression.control()
> 
> init=rrl.NonLinearRegression.initdata()

We  supply a Jacobi matrix, a function and an initial guess for the parameters to be found (assuming the function has just 2 curve fitting parameters):

> init.Jacobian=Jacobi
>
> init.f=linear
> 
> init.initialguess = [1,1]

Where  Jacobi and linear are two user defined functions. 

If we would want to fit a line, we would have to implement the function f(X,beta) and the jacobian J(X,beta) as follows

>def linear(X, beta):
>
> &emsp;   Y=[]
>
> &emsp;   for i in range(0,len(X)):
>
> &emsp;   &emsp;    Y.append(beta[0] * X[i] + beta[1])
>
> &emsp;    return Y


> def Jacobi(X, beta):
> 
>&emsp;	m=rrl.MatrixCode.Matrix (len(X), len(beta))
> 
>&emsp;	for i in range(0,len(X)):
> 
>&emsp;	&emsp;	m[i, 0] = X[i]
> 
>&emsp;&emsp;	m[i, 1] = 1
> 
>&emsp;	return m

Then we can call the Levenberg-Marquardt algorithm

> rrl.NonLinearRegression.non_linear_regression(X2, Y2, init, ctrl, res)

and then print the result:
> print("Slope")
> 
> print(res.beta[0])
> 
> print("Intercept")
> 
> print(res.beta[1])

The class NonLinearRegression.control has various parameters that control the behavior of the Levenberg-Marquardt algorithm, among them are various
conditions when the algorithm should stop (e.g. after some time, or after there is no improvement or after the error is below a certain margin). 
They may be set as desired.

Additionally, there are parameters that control the step size of the algorithm. These parameters have the same names as described at
https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm and if not otherwise specified,
defaults are used for them usually work.


### Non-linear robust regression
For non-linear curve fits the library also has the modified forward search/modified lts algorithm and the iterative outlier removal as for linear regression.
For a modified forward search/lts algorithm with default parameters (S estimator, outlier_tolerance=3, loss function as least squares, 30% of the points are outliers at maximum), a call looks as follows:

First the initialisation :
> res=rrl.RobustRegression.nonlinear_algorithm_result()
> 
> ctrl=rrl.RobustRegression.modified_lts_control_nonlinear()
> 
> init=rrl.NonLinearRegression.initdata()
> 
> init.Jacobian=Jacobi
> 
> init.f=linear
> 
> init.initialguess = [1,1]

Then the function call:
> rrl.RobustRegression.modified_lts_regression_nonlinear(X2, Y2, init, ctrl, res)

Finally, we print the result:
> print("Slope")
> 
> print(res.beta[0])
> 
> print("Intercept")
> 
> print(res.beta[1])
> 
> print("\nOutlier indices")
> 
> for ind in res.indices_of_removedpoints:
> 
> &emsp;  print(ind)




For the interative outlier removal algorithm, a call to the regression function with default parameters (S estimator, outlier_tolerance=3, loss function as least squares, 30% of the points are outliers at maximum) would look as follows:

> res=rrl.RobustRegression.nonlinear_algorithm_result()
> 
> ctrl=rrl.RobustRegression.nonlinear_algorithm_control()
> 
> init=rrl.NonLinearRegression.initdata()
> 
> init.Jacobian=Jacobi
> 
> init.f=linear
> 
> init.initialguess = [1,1]

> rrl.RobustRegression.iterative_outlier_removal_regression_nonlinear(X2, Y2, init, ctrl, res)

### Custom error functions
By default, the library uses the sum of the squared residuals divided by the pointnumber as a loss function.
One can also specify Huber's loss function, but then one also has to supply a border parameter beyond which the loss function starts its linear behavior.
One also can set a log cosh loss function, or a quantile loss function. The latter needs a gamma parameter to be specified within the interval of 0 and 1.

Finally, one can define a custom loss function with a callback mechanism.

We may define user defined loss functions. This is done in two steps. A function

> def err_pp(Y,fY,pointnumber):
> &emsp;   return (Y-fY)*(Y-fY)/pointnumber

Computes a residual between the data and the curve fit for a single point.
Another function

> def aggregate_err(errs):
> &emsp;    res=0 
> &emsp;   for i in range(0,len(errs)):
> &emsp;  &emsp;     res+=errs[i]
> &emsp;   return res

computes an entire error from a list of residuals generated by the function err_pp.
Note that if the data is such that it does not correspond perfectly to the curve, this should at best be some kind of average error instead of a simple sum.
Since otherwise, removing a point will always reduce the error. Since the robust methods delete points based on the aggregate error, this would usually lead to
curve fits which do not have enough points taken into consideration. The division by the pointnumber can be done in err_pp (as in this example) or in aggregate_err.

The following call will then make a robust curve fit with the custom error function

> res9=rrl.RobustRegression.nonlinear_algorithm_result() 
> ctrl9=rrl.RobustRegression.modified_lts_control_nonlinear()
> ctrl9.lossfunction=rrl.LossFunctions.custom
> ctrl9.loss_perpoint=err_pp
> ctrl9.aggregate_error=aggregate_err

> init9=rrl.NonLinearRegression.initdata() 
> init9.Jacobian=Jacobi
> init9.f=linear
> init9.initialguess = [1,1]
> rrl.RobustRegression.modified_lts_regression_nonlinear(X2, Y2, init9, ctrl9, res9)

Note that if we use the linear versions of the robust regression, then these methods would just make simple linear
fits or repeated median fits, which minimize their own loss function, and the selected loss function of the library
is then only used for outlier removal.

With the robust non-linear regression algorithms, the custom error functions are used for the curve fits as well 
as for the outlier removal procedures.

If one needs a linear fit where the custom error function is used as a metric for the curve fit as well as for the outlier
removal, one has to use the non-linear algorithm with a linear call back function. 

Note also that the quantile loss function is asymmetric. Therefore, the quantile loss function should mostly be used with
the linear robust curve fitting algorithms, since then it is only used for outlier removal. 
If the quantile loss function is used with the non-linear robust algorithms it is likely to confuse the Levenberg-Marquardt algorithm 
because of its asymmetry.




## For the C++ language:

In general, one has to include the library headers as follows:

> #include "statisticfunctions.h"
>
> #include "Matrixcode.h"
> 
> #include "linearregression.h"
> 
> #include "robustregression.h"
> 
> #include "nonlinearregression.h"
>
> #include "lossfunctions.h" 
>
> #include <valarray>

### Simple Linear Regression
The usage of the library in C++ is essentially similar as in Python. the testapplication.cpp demonstrates the same function calls.
The the X and Y data are stored in C++ valarrays. The control, result and initdata are not classes, but structs.

For example, if we define some X,Y data:

> valarray<double> X = { -3, 5,7, 10,13,16,20,22 };
> 
> valarray<double> Y = { -210, 430, 590,830,1070,1310,1630,1790 };

and initialize the struct where we store the result,

> Linear_Regression::result res;

we can call a linear regression as follows:

> Linear_Regression::linear_regression(X, Y, res);

and we may print the result:

> printf(" Slope ");
> 
> printf("%f", res.main_slope);
> 
> printf("\n Intercept ");
> 
> printf("%f", res.main_intercept);

### Robust Regression methods

Let us first define X and Y data with two outliers added.
>	valarray<double> X2 = { -3, 5,7, 10,13,15,16,20,22,25 };
>
>	valarray<double> Y2 = { -210, 430, 590,830,1070,20,1310,1630,1790,-3 };

#### Median Linear Regression: 
Median regression is slower but more robust as simple linear regression.
This command calls a robust curve fit with median regression

> Linear_Regression::result res;
> 
> Linear_Regression::median_linear_regression(X2, Y2, res);

and then we print the result

> printf(" Slope ");
> 
> printf("%f", res.main_slope);
> 
> printf("\n Intercept ");
> 
> printf("%f", res.main_intercept);

#### Modified forward search
When many and large outliers are present, median regression does sometimes not deliver the desired results.

The library therefore has the modified forward search/modified lts algorithm and the iterative outlier removal algorithm that can
find outliers and remove them from the curve fit entirely.

Below is a call for a robust modified forward search/modified lts algorithm initialized with default
parameters:

First the structs for controlling the algorithm and storing the results are initialised,

> Robust_Regression::modified_lts_control_linear ctrl;
> 
> Robust_Regression::linear_algorithm_result res;

Then, we call the functions for the curve fit:
>	Robust_Regression::modified_lts_regression_linear(X2, Y2, ctrl, res);

Then we print the result:

>	printf(" Slope ");
>
>	printf("%f", res.main_slope);
>
>	printf("\n Intercept ");
>
>	printf("%f", res.main_intercept);
>
>	printf("\n Indices of outliers ");
>
>	for (size_t i = 0; i < res.indices_of_removedpoints.size(); i++)
>	{
>	&emsp; 	size_t w = res.indices_of_removedpoints[i];
>
>	&emsp;	printf("%lu", (unsigned long)w);
>
>	&emsp;	printf(", ");
>
>	}


The default parameters are as follows:

The S estimator is used, outlier_tolerance=3,  30% of the pointnumber are outliers at maximum, loss function is given by least squares of the residuals.

As in the Python documentation, a point with residual err is then an outlier if 

> |err-median(errs)/S_estimator>3

where errs is the array of residuals.


If the Q-estimator is used instead, the initialisation for the modified forward search/lts algorithm looks like

>	Robust_Regression::modified_lts_control_linear ctrl;
>
>	Robust_Regression::linear_algorithm_result res;
>
>	ctrl.rejection_method = Robust_Regression::tolerance_is_decision_in_Q_ESTIMATION;

Then we call the regression function:
>	Robust_Regression::modified_lts_regression_linear(X2, Y2, ctrl, res);

If the interquartile range estimator should be used, so that a point is removed if it is below Q1 − outlier_tolerance* IQR or above Q3 + outlier_tolerance IQR, 
we would have to set:

> ctrl.rejection_method = Robust_Regression::tolerance_is_interquartile_range;

For the interquartile range estimator, outlier_tolerance should be set to 1.5, so we additionally have to write:
>	ctrl.outlier_tolerance = 1.5;

before we call the regression function:
>	Robust_Regression::modified_lts_regression_linear(X2, Y2, ctrl, res);

Similarly, some may prefer to set the outlier_tolerance parameter to 3.5 when the S,Q, or MAD or other estimators are used.

The loss function can also be changed. For example, the absolute value |err| of the residuals is usually more robust than the square err^2.
The command

> ctrl.lossfunction = LossFunctions::absolutevalue;

changes the lossfunction to the absolute value. 

One can also specify Huber's loss function, but then one also has to supply a border parameter  beyond which the function starts its linear behavior.
One also can set a log cosh loss function, or a quantile loss function. The latter needs a gamma parameter to be specified within the interval of 0 and 1.
Finally, one can define a custom loss function with a callback mechanism.


Note that if we use the linear versions of the robust regression, then these methods would just make simple linear
fits or repeated median fits, which minimize their own loss function, and the selected loss function of the library
is then only used for outlier removal.

With the robust non-linear regression algorithms, the custom error functions are used for the curve fits as well 
as for the outlier removal procedures.

If one needs a linear fit where the custom error function is used as a metric for the curve fit as well as for the outlier
removal, one has to use the non-linear algorithm with a linear call back function. 

Note also that the quantile loss function is asymmetric. Therefore, the quantile loss function should mostly be used with
the linear robust curve fitting algorithms, since then it is only used for outlier removal. 
If the quantile loss function is used with the non-linear robust algorithms it is likely to confuse the Levenberg-Marquardt algorithm 
because of its asymmetry.

#### Iterative outlier removal
The modified forward search/modified lts algorithm can be slow since its complexity is given by the binomial coefficient of the pointnumber over the maximum
number of outliers to be found. The iterative outlier removal algorithm is faster. It makes a curve fit of all points, then removes the outliers based on the 
loss function and estimator and makes another curve fit and repeats the procedure until no outliers are found.

A call to this method with the default parameters S estimator, outlier_tolerance=3, the loss function as least_squares and at most 30% of the points
designated as outliers, would look as follows:

>	Robust_Regression::linear_algorithm_result res;
>
>	Robust_Regression::linear_algorithm_control ctrl;
>
>	Robust_Regression::iterative_outlier_removal_regression_linear(X2, Y2, ctrl, res);

### Non-linear Regression
The non-linear curve fitting algorithm implements a Levenberg-Marquardt algorithm.

For it to work, we must first supply initialisation data in form of a valarray of initial parameters beta,
a function f(X,beta) to be found and a Jacobian J(X,beta).

if we wanted to fit a line, we would have do define the function f(X,beta) to be fit

> valarray<double> linear(const valarray<double>&X,const  valarray<double>& beta)
> {
> 
>	&emsp;valarray<double> Y(X.size());
> 
>	&emsp;for (size_t i = 0; i < X.size(); i++)
> 
>	&emsp;&emsp;	Y[i] = beta[0] * X[i] + beta[1];
> 
>	&emsp;return Y;
> 
> }
>

and its Jacobian Matrix J(X,beta)
> Matrix Jacobi(const valarray<double>& X, const  valarray<double>& beta)
> {
> 
>	&emsp;Matrix ret(X.size(), beta.size());
> 
>	&emsp;for (size_t i = 0; i < X.size(); i++)
> 
>	&emsp;{
> 
>	&emsp;&emsp;	ret(i, 0) = X[i];
> 
>	&emsp;&emsp;	ret(i, 1) = 1;
> 
>	&emsp;}
> 
>	&emsp;return ret;
> 
> }

A non-linear fit can the be initialised with default control parameters for the Levenberg-Marquardt algorithm like this:


>	Non_Linear_Regression::result res;
>
>	Non_Linear_Regression::control ctrl;
>
>	Non_Linear_Regression::initdata init;
>
>	init.f = linear;
>
>	init.J = Jacobi;
>
>	valarray<double>beta = { 1,1 };
>
>	init.initialguess = beta;

Then one can call the function:
>	Non_Linear_Regression::non_linear_regression(X, Y, init, ctrl, res);

Then we may print the result:

>	printf("\n Slope ");
>
>	printf("%f", res.beta[0]);
>
>	printf("\n intercept ");
>
>	printf("%f", res.beta[1]);

The struct Non_Linear_Regression::control has various control parameters that control the behavior of the Levenberg-Marquardt algorithm, among them are various
conditions when the algorithm should stop (e.g. after some time, or after there is no improvement or after the error is below a certain margin).
They may be set as desired.

Additionally, there are parameters that control the step size of the algorithm. These parameters have the same names as described at
https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm and if not otherwise specified,
defaults are used for them  usually work.


### Non-linear robust curve fits

As for the linear regression, the library has the same modified forward search/lts and iterative outlier removal algorithms for the non-linear case

For a modified forward search/lts algorithm, a call looks as follows:

First the initialisation, here with default parameters for the algorithm control:

> Robust_Regression::modified_lts_control_nonlinear ctrl;
> 
> Robust_Regression::nonlinear_algorithm_result res;
> 
> Non_Linear_Regression::initdata init;
> 
> init.f = linear;
> 
> init.J = Jacobi;
> 
> valarray<double>beta = { 1,1 };
> 
> init.initialguess = beta;

Then the call:

> Robust_Regression::modified_lts_regression_nonlinear(X2, Y2, init, ctrl, res);

Then we print the result:
> 	printf(" Slope ");
> 
>	printf("%f", res.beta[0]);
> 
>	printf("\n Intercept ");
>
>	printf("%f", res.beta[1]);
> 
>	printf("\n Indices of outliers ");
>
>	for (size_t i = 0; i < res.indices_of_removedpoints.size(); i++){
>	&emsp;	size_t w = res.indices_of_removedpoints[i];
>  	&emsp;	printf("%lu", (unsigned long)w);
>   &emsp;  printf(", ");
>	}


For the interative outlier removal algorithm, the call to the regression function would look as follows:
First the initialisation, here again with default parameters for the algorithm control:

> Non_Linear_Regression::initdata init;
> 
> Robust_Regression::nonlinear_algorithm_control ctrl;
> 
> init.f = linear;
> 
> init.J = Jacobi;
> 
> valarray<double>beta = { 1,1 };
> 
> init.initialguess = beta;

Then the call,

> Robust_Regression::iterative_outlier_removal_regression_nonlinear(X2, Y2, init, ctrl, res);

The printing of the result would work similar as above.

### Custom error functions
By default, the library uses the sum of the squared residuals divided by the pointnumber as a loss function.
One can also specify Huber's loss function, but then one also has to supply a border parameter beyond which the loss function becomes linear.
One also can set a log cosh loss function, or a quantile loss function. The latter needs a gamma parameter to be specified within the interval of 0 and 1.

Finally, one can define a custom loss function with a callback mechanism.

We may define user defined loss functions. This is done in two steps. A function

> double err_pp(const double Y, double fY, const size_t pointnumber) {
> &emsp;	return ((Y - fY)* (Y - fY)) /(double) pointnumber;
> }

Computes a residual between the data and the curve fit for a single point.

Another function

> double aggregate_err(valarray<double>& err){
> &emsp;	return err.sum();
> }

computes an entire error from a list of residuals generated by the function err_pp.
Note that if the data is such that it does not correspond perfectly to the curve, this should at best be some kind of average error instead of a simple sum.
Since otherwise, removing a point will always reduce the error. Since the robust methods delete points based on the aggregate error, this would usually lead to
curve fits which do not have enough points taken into consideration. The division by the pointnumber can be done in err_pp (as in this example) or in aggregate_err.

The following will then make a robust curve fit with the custom error function:
At first the usual initialisation
> Non_Linear_Regression::initdata init13;
>	init13.f = linear;
>	init13.J = Jacobi;
>	init13.initialguess = beta;
>  Robust_Regression::nonlinear_algorithm_result res13;

Then the set of the custom loss function

> Robust_Regression::modified_lts_control_nonlinear ctrl13;
> ctrl13.lossfunction = LossFunctions::custom;
> ctrl13.loss_pp = err_pp;
> ctrl13.agg_err = aggregate_err;

Note that if the aggregate error would not be defined here, the results of the calls of the loss functions per point would just be summed.

Finally, we can make the function call	

> Robust_Regression::modified_lts_regression_nonlinear(X2, Y2, init13, ctrl13, res13);

Note that if we use the linear versions of the robust regression, then these methods would just make simple linear
fits or repeated median fits, which minimize their own loss function, and the selected loss function of the library
is then only used for outlier removal.

With the robust non-linear regression algorithms, the custom error functions are used for the curve fits as well 
as for the outlier removal procedures.

If one needs a linear fit where the custom error function is used as a metric for the curve fit as well as for the outlier
removal, one has to use the non-linear algorithm with a linear call back function. 

Note also that the quantile loss function is asymmetric. Therefore, the quantile loss function should mostly be used with
the linear robust curve fitting algorithms, since then it is only used for outlier removal. 
If the quantile loss function is used with the non-linear robust algorithms it is likely to confuse the Levenberg-Marquardt algorithm 
because of its asymmetry.

# Further documentation
The library has an online repository at https://github.com/bschulz81/robustregression where the source code can be accessed. 

The detailed documentation of all the various control parameters of the curve fiting algorithms is in the docstrings of the Python module and in the c++ header file. 

Also, the C++/Python test applications in the folder testapp are documented and show many function calls

