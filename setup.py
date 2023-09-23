"""
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
"""

from skbuild import setup 
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
name="pyRobustRegressionLib",
version="1.2.1",
description="A library that implements algorithms for linear and non-linear robust regression.",
long_description=long_description,
long_description_content_type='text/markdown',
author='Benjamin Schulz',
license="MIT License",
packages=['pyRobustRegressionLib'],
python_requires=">=3.7",
package_dir={"": "library\src"},
keywords="""robust regression, forward-search, Huber\'s loss functtion, median regression, 
simple linear regression, non-linear regression, Levenberg-Marquardt algorithm, 
statistics, estimators,  S-estimator, Q-estimator, Student t-distribution,
interquartile range, machine learning""",
url='https://github.com/bschulz81/robustregression',
classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers"
    ]
)
