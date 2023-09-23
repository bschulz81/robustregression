# robustregression
This is a c++ library with statistical machine learning algorithms for linear and non-linear robust regression.

It implements the statistical algorithms that were originally developed by the author for an autofocus application for telescopes

and published in 	arXiv:2201.12466 [astro-ph.IM], https://doi.org/10.1093/mnras/stac189

In addition to these, two other robust algorithms were added and the curve fitting library has been brought into a form of a
clear and simply API that can be easily used for very broad and general applications.

The library offers python bindings for most functions. So the programmer has the choice between c++ and python. In order to 
compile the library with python bindings Pybind11 and Python3 should be installed and be found by CMake. 
Otherwise, only the C++ standard template library is used, together with OpenMP. 

The documentation of the functions that the library uses are written in the C++header files and in the __doc__ methods of the python bindings.

In addition, a c++ application and a python script is provided that show the functions of the library with very simple data.

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
9. Croux, C., Rousseeuw, P.J. (1992). Time-Efficient Algorithms for Two Highly Robust Estimators of Scale. In: Dodge, Y., Whittaker, J. (eds) Computational Statistics. Physica, Heidelberg. https://doi.org/10.1007/978-3-662-26811-7_58 (For the faster version of the S and Q-estimators.) The versions of the S and Q estimators in this library are now adapted from the algorithms of Croux and Rousseeuw to the C language. Note that it is not the same Code because of some optimizations. Since many variables act on array indices in these algorithms, they were actually non-trivial to convert from Fortran to C. For the Q estimator, the naive algorithm is faster for less than 100 datapoints. For the S estimator this is the case for less than 10 datapoints. Therefore, in these cases the naive versions are still used.

# Compiling and Installing the library:

The Library needs CMake and a C compiler that is at least able to generate code according to the C14 standard (per default, if one does not use Clang or MacOs, it uses C17, but with a CXX_STANDARD 14 switch set in the CMakeLists.txt for the library, it can use C14, which is the the default for Clang and MacOS.) 

The library also makes use of OpenMP and needs Python in version 3 and pybind11 to compile. 


Per default, the CMake variable $WithPython is ON. If one wants to use the python module one can compile and install the module by 
with pip, i.e. by typinh

> pip install .

in the package directory. After that, the binary extension is copied into a path for python wheel extensions, where python scripts can find it.

In addition, a build directory should appear where the c++ testapplication can be found together with a binary.

one can uninstall the module by typing


> pip uninstall pyRobustRegressionLib



If one does not want the python module, one may set the cmake variable WithPython to OFF.

One can compile the library also traditionally via CMake. Typing 

> CMake . 

in the package directory will generate the files necessary to compile the library, depending on the CMake generator set by the user.

After compilation, an out directory will appear with the library in binary form and the executable testapplication in c++. 
If the variable WithPython was set to ON, one will also find a python module and a test script in python. 

# Documentation of the library functions
Documentation of the API is provided in C++ header files in the /library/include directory and the docstrings for the python module in the src/pyRobustRegressionLib
Module. The latter It can be loaded in python scripts with 

> import pyRobustRegressionLib as rrl

The command 

> print(rrl.\__doc__)

Will list the sub-modules of the library, which are 

- StatisticFunctions, 
- LinearRegression, 
- MatrixCode, 
- NonLinearRegression and 
- RobustRegression

And their docstrings can be called e.g. by
>print(rrl.*SubModuleName*.\__doc__)

e.g.

> print(rrl.StatisticFunctions.\__doc__).

Will list the functions and classes of the sub-module StatisticFunctions. The free functions and classes all have more detailed doc
strings that can be called as below for example

> print (rrl.MatrixCode.Identity.\__doc__)

More convenient documentation is provided in the header files of the C++ source code of the package,
which can be found in the /library/include directory.

The header files can be found in the include subdirectory of the package.

In the testapp folder, two example programs, one in python and one in C++ is provided.
These test applications have extensive comments and call many functions of the library in order to show the basic usage. 

The curve fits that are done in the provided example programs are, however, very simple of course.
This was done in order to keep the demonstration short and simple.
The library is of course intended to be used with larger and much more complicated data.

#

The library has an online repository at https://github.com/bschulz81/robustregression where the source code can be accessed. 
