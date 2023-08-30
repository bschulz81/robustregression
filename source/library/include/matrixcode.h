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
#include <stdint.h>
#include "robustregressionlib_exports.h"

using namespace std;

//vector class
//provides common operators for vector computation.
class ROBUSTREGRESSION_API Vector
{
public:
	  // initializes a vector with size s and all elements set to 0.0 
	  Vector(const size_t s);
	  //initializes a vector with the elements of a valarray. vector(i)=v[i].
	  Vector(valarray<double> v);
	  Vector(valarray<double> *v);

	  //operators for typical vector calculations
	  Vector operator+(const Vector&)const;
	  Vector operator-(const Vector&)const;
	  double operator*(const Vector&)const;
	  Vector operator*(const double&)const;
	  Vector operator/(const double&)const;
	  double& operator()(const size_t);
	  const double& operator()(const size_t)const;
	  inline operator valarray<double>() const { return m; }
	  Vector operator=( Vector&);
	  Vector operator=( Vector*); 
	  //returns the size of a vector
	  size_t Size() const;
	  //resizes the vector and sets its size to 0.
	  void Resize(const size_t t);
private:
	  valarray<double> m;
};


//Matrix class, provides common operators for n x n dimenstional matrices.
class ROBUSTREGRESSION_API Matrix
{
public:
	// initializes a matrix with given rows and columns and values zero.
	  Matrix(const size_t rows, const size_t columns);
	  //initializes a matrix with given rows and columns and fills it with the values of a given array, 
	  // the value of Matrix(row,column) should be given by m1[row * columns + column]
	  Matrix(const size_t rows, const size_t columns, valarray<double> m1);
	  Matrix(const size_t rows, const size_t columns, valarray<double> *m1);

	  //common operators vor matrix calculations
	  Matrix operator+(const Matrix&)const;
	  Matrix operator-(const Matrix&)const;
	  Matrix operator*(const Matrix&)const;
	  Vector operator*(const Vector&)const;
	  valarray<double> Matrix::operator*(const valarray<double>& B)const;
	  Matrix operator*(const double&)const;
	  Matrix operator/(const double&)const;
	  double& operator()( const size_t row, const  size_t column) ;
	  const double& operator()(const size_t row, const  size_t column)const;
	  inline operator valarray<double>() const {  return m;  }

	  // returns the number of rows
	  size_t Rows() const;
	  //returns the number of columns
	  size_t Columns() const;
	  //resizes a matrix and sets its values to zero
	  void Resize(const size_t rows, const size_t columns);
	  //interchanges rows of a matrix.
	  void SwapRows(const size_t row1, const size_t row2);
private:
	size_t r;
	size_t c;
	valarray<double> m;
};


// provides common functions for matrices.

namespace Matrixcode
{
	//solves a linear equation. expects a matrix and a vector and returns a vector.
	ROBUSTREGRESSION_API Vector Gaussian_algorithm(const Matrix &m, const Vector& v);
	ROBUSTREGRESSION_API valarray<double> Gaussian_algorithm(const Matrix &m, const  valarray<double>& v);
	//yields the transpose of a matrix
	ROBUSTREGRESSION_API Matrix Transpose(const Matrix& m);
	//yields an identity matrix
	ROBUSTREGRESSION_API Matrix Identity(const size_t rows, const size_t columns);
	//yields the diagonal of a matrix, with all other entries put to zero.
	ROBUSTREGRESSION_API Matrix Diagonal(const Matrix& m);


	//prints a matrix
	ROBUSTREGRESSION_API void printmatrix(const Matrix& m);
	//prints a vector
	ROBUSTREGRESSION_API void printvector(const Vector& v);
}
