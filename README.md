# robustregression
This is a c++ library with statistical machine learning algorithms for linear and non-linear robust regression.

It implements the statistical algorithms that were originally developed by the author for an autofocus application for telescopes

and published in 	arXiv:2201.12466 [astro-ph.IM], https://doi.org/10.1093/mnras/stac189

In addition to these, two other robust algorithms were added and the curve fitting library has been brought into a form of a
clear and simply API that can be easily used for very broad and general applications.

The library offers python bindings for most functions. So the programmer has the choice between c++ and python. In order to 
compile the library with python bindings Pybind11 and Python3 should be installed and be found by CMake. 
Otherwise, only the C++ standard template library is used, together with OpenMP. 

The documentation of the functions that the library uses are written in the header files.

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
9. Croux, C., Rousseeuw, P.J. (1992). Time-Efficient Algorithms for Two Highly Robust Estimators of Scale. In: Dodge, Y., Whittaker, J. (eds) Computational Statistics. Physica, Heidelberg. https://doi.org/10.1007/978-3-662-26811-7_58 (For the faster version of the S-estimator.) The version of the S estimator in this library now is adapted from Croux and Rousseeuw to the C language. Note that it is not the same Code because of some optimizations. Since many variables act on array indices in this algorithm, it was actually non-trivial to convert from Fortran to C.

# Compiling and Installing the library:
The Library needs CMake and a C compiler that is at least able to generate code according to the C14 standard (per default, it uses C17, but with a switch in the CMakeLists.txt for the library, it can use C14. 
It also makes use of OpenMP and needs Python in version 3 and pybind 11 to compile. After compilation with CMake, an /out folder appears where the library and the compiled test applications and a python test script can be found which demonstrate how to use the library.

