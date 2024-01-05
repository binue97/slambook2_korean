#include <iostream>

using namespace std;

#include <ctime>
// Eigen core components
#include <Eigen/Core>
// Algebraic operations of dense matrix
#include <Eigen/Dense>

using namespace Eigen;

#define MATRIX_SIZE 50

/****************************
* This program demonstrates the use of Eigen basic types.
****************************/

int main(int argc, char **argv) {
  // All vectors and matrices in Eigen are Eigen::Matrix which is a template class.
  // Matrix<DataType, Row, Col>
  Matrix<float, 2, 3> matrix_23;

  // Eigen provides many built-in types via typedef, but the base layer is still Eigen::Matrix.
  // For example, Vector3d is essentially Eigen::Matrix<double, 3, 1>, which is a three-dimensional vector.
  Vector3d v_3d;
  Matrix<float, 3, 1> vd_3d;

  // Matrix3d is essentially Eigen::Matrix<double, 3, 3>
  Matrix3d matrix_33 = Matrix3d::Zero(); // Initialize with zero.
  // If you are unsure of the matrix size, you can use a dynamically sized matrix.
  Matrix<double, Dynamic, Dynamic> matrix_dynamic;
  // Simpler form using typedef
  MatrixXd matrix_x;

  // Initialization using operator overloading
  matrix_23 << 1, 2, 3, 4, 5, 6;
  cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << endl;

  // Access matrix element with operator ()
  cout << "print matrix 2x3: " << endl;
  for (int i = 0; i < 2; i++) 
  {
    for (int j = 0; j < 3; j++) 
     cout << matrix_23(i, j) << "\t";
    cout << endl;
  }

  // Initialize vectors declared above.
  v_3d << 3, 2, 1;
  vd_3d << 4, 5, 6;

  // When operating with matrix and vectors, their data type should match.
  // Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d; <-- double and float can't be operated together.
  // So, user must explicitly cast their data types.
  Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
  cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << endl;

  Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
  cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << endl;

  // When multiplying matrices, user should be aware of their dimensions.
  // Wrong dimensions will trigger error.
  // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

  // Some matrix operations
  matrix_33 = Matrix3d::Random();
  cout << "random matrix: \n" << matrix_33 << endl;
  cout << "transpose: \n" << matrix_33.transpose() << endl;
  cout << "sum: " << matrix_33.sum() << endl; // Sum of all elements in matrix.
  cout << "trace: " << matrix_33.trace() << endl;
  cout << "times 10: \n" << 10 * matrix_33 << endl;
  cout << "inverse: \n" << matrix_33.inverse() << endl;
  cout << "det: " << matrix_33.determinant() << endl;

  // Eigen value
  // The real symmetric matrix can guarantee the diagonalization success.
  SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
  cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
  cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

  // Solving equations
  // matrix_NN * x = v_Nd
  // MATRIX_SIZE is defined by macro, and matrix_NN is initialized with random value.
  // Directly calculating inverse matrix is straight-forward, but it is a lot of computation.
  Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
      = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
  matrix_NN = matrix_NN * matrix_NN.transpose();  // Semi-positive definite
  Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

  clock_t time_stt = clock();
  // Direct inversion
  Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
  cout << "time of normal inverse is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  // Usually used matrix decomposition, such as QR decomposition, will be much faster.
  time_stt = clock();
  x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
  cout << "time of Qr decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  // For positive definite matrices, the equation can also be solved by Cholesky decomposition.
  time_stt = clock();
  x = matrix_NN.ldlt().solve(v_Nd);
  cout << "time of ldlt decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  return 0;
}