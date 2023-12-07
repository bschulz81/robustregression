1.3.1

Included a fix that ensures pybind11 is found


1.3.0

The library now compiles sucessfully under linux

The test applications can be run under linux

Made extensive use of Open-MP simd instructions whenever possible now.

Added the Loss functions: logcosh, quantile, custom, 
custom uses a call back mechanism where the user can define own loss functions.

Documented the new loss functions in the header, the python bindings, as well as the Readme.MD

Documented the use of the custom loss function in the example programs

Updated the README.MD

Fixed a spelling error of a docstring in the python-bindings file.

Updated the documentation of the lowmedian function

Fixed the documentation of Linear and reoeated Median Regression in the Python file

Added literature references for the repeated median fit in the header files, the python bindings and the README.md

Added a Changelog.md file

