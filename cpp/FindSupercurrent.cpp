#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

#include <vector>
#include <array>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <complex>
#include <cmath>
#include <bit>
#include <chrono>
#include <limits>



///// Transformation matrix from complex spinor to real spinor

Eigen::MatrixXcd makeCpxSpinorToRealSpinor() {
    using namespace std::complex_literals;

    // the 2x2 block
    Eigen::Matrix2cd block;
    block << 1.0, -1.0i,
            1.0, 1.0i;

    // the 24x24 zero matrix
    Eigen::MatrixXcd mat = Eigen::MatrixXcd::Zero(24, 24);

    // Place the 2x2 block diagonally
    for (int k = 0; k < 12; k++) {
        mat.block<2,2>(2*k, 2*k) = block;
    }

    // Normalize
    mat *= 1.0 / std::sqrt(2.0);

    return mat;
}

Eigen::MatrixXcd CpxSpinorToRealSpinor = makeCpxSpinorToRealSpinor();



///// the Clifford algebra action on spinors

std::vector<std::vector<std::array<double,2>>> PsiActionList(24, std::vector<std::array<double,2>>(4096)); // loaded later in main()

Eigen::VectorXcd PsiAction(int i, const Eigen::VectorXcd& spinor) {
    int dim = 1 << 12; // 2^12
    Eigen::VectorXcd result = Eigen::VectorXcd::Zero(dim);
    for (int d = 0; d < dim; d++) {
        int index = static_cast<int>(PsiActionList[i][d][0]);
        double coeff = PsiActionList[i][d][1];
        result[d] = coeff * spinor[index];
    }
    return result;
}

Eigen::VectorXcd PsiLinCombiAction(const Eigen::VectorXcd& coeffList,
                                   const Eigen::VectorXcd& spinor) {
    int dim = spinor.size();
    Eigen::VectorXcd result = Eigen::VectorXcd::Zero(dim);
    int terms = coeffList.size();
    for (int i = 0; i < terms; i++) {
        result += coeffList[i] * PsiAction(i, spinor);
    }
    return result;
}

Eigen::VectorXcd SO24LiftAction(const std::vector<double>& angles,
                                const Eigen::MatrixXd& eigenmat,
                                const Eigen::VectorXcd& spinor) {
    Eigen::MatrixXcd eigenmatCpx = CpxSpinorToRealSpinor * eigenmat;
    Eigen::VectorXcd temp = spinor;

    for (int i = 12; i >= 1; i--) {
        Eigen::VectorXcd cosTemp = std::cos(angles[i-1] * M_PI) * temp;

        Eigen::VectorXcd coeffList1 = eigenmatCpx.col(2*i-1);
        Eigen::VectorXcd coeffList2 = eigenmatCpx.col(2*i-2);

        temp = PsiLinCombiAction(coeffList1, temp);
        temp = PsiLinCombiAction(coeffList2, temp);

        temp = std::sin(angles[i-1] * M_PI) * temp;
        temp = cosTemp + temp;
    }

    return temp;
}



///// Retain only spinors with positive chirality (against expectations, a bit slower than above)

std::vector<std::vector<std::array<double,2>>> PsiPsiActionList(276, std::vector<std::array<double,2>>(2048));

Eigen::VectorXcd PsiPsiAction(int x, const Eigen::VectorXcd& spinorChi) {
    int dim = 1 << 11; // 2^11
    Eigen::VectorXcd result = Eigen::VectorXcd::Zero(dim);
    for (int d = 0; d < dim; d++) {
        int index = static_cast<int>(PsiPsiActionList[x][d][0]);
        double coeff = PsiPsiActionList[x][d][1];
        result[d] = coeff * spinorChi[index];
    }
    return result;
}

Eigen::VectorXcd PsiPsiLinCombiAction(const Eigen::VectorXcd& PsiPsiCoeffList,
                                   const Eigen::VectorXcd& spinorChi) {
    int dim = spinorChi.size();
    Eigen::VectorXcd result = Eigen::VectorXcd::Zero(dim);
    int terms = PsiPsiCoeffList.size();
    for (int i = 0; i < terms-1; i++) {
        result += PsiPsiCoeffList[i] * PsiPsiAction(i, spinorChi);
    }
    result += PsiPsiCoeffList[terms-1] * spinorChi; // constant term
    return result;
}

// Multiplication of two linear combinations of Psi's
Eigen::VectorXcd MultiOfPsiLinCombi(const Eigen::VectorXcd& coeffList1,
                                    const Eigen::VectorXcd& coeffList2) {
    int n = 24;
    Eigen::VectorXcd result((n*(n-1))/2 + 1); // number of PsiPsi's + constant term
    int idx = 0;

    // coefficient of Psi_i Psi_j
    for (int i = 0; i < n-1; ++i) { // i = 0..22
        for (int j = i+1; j < n; ++j) { // j = i+1..23
            result(idx) = coeffList1(i) * coeffList2(j) - coeffList1(j) * coeffList2(i);
            idx++;
        }
    }

    // constant term: -2 * Sum[coeffList1[[2 i]] coeffList2[[2 i -1]], {i, 1, 12}] in Mathematica (1-indexed)
    std::complex<double> constant = 0.0;
    for (int i = 0; i < 12; ++i) {
        constant += coeffList1(2*i + 1) * coeffList2(2*i);
    }
    constant *= -2.0;
    result(idx) = constant;

    return result;
}

Eigen::VectorXcd SO24LiftActionChi(const std::vector<double>& angles,
                                   const Eigen::MatrixXd& eigenmat,
                                   const Eigen::VectorXcd& spinorChi) {
    Eigen::MatrixXcd eigenmatCpx = CpxSpinorToRealSpinor * eigenmat;
    Eigen::VectorXcd temp = spinorChi;

    for (int i = 12; i >= 1; --i) {
        Eigen::VectorXcd cosTemp = std::cos(angles[i-1] * M_PI) * temp;

        Eigen::VectorXcd coeffList1 = eigenmatCpx.col(2*i - 2);
        Eigen::VectorXcd coeffList2 = eigenmatCpx.col(2*i - 1);

        Eigen::VectorXcd multiCoeffs = MultiOfPsiLinCombi(coeffList1, coeffList2);

        temp = PsiPsiLinCombiAction(multiCoeffs, temp);

        temp = std::sin(angles[i-1] * M_PI) * temp;
        temp = cosTemp + temp;
    }

    return temp;
}

// Make positive-chiral spinor into full spinor
Eigen::VectorXcd ChiSpinToFullSpin(const Eigen::VectorXcd& ChiSpin) {
    int dim = 1 << 12; // 2^12
    Eigen::VectorXcd result = Eigen::VectorXcd::Zero(dim);
    for (int d = 0; d < dim; d++) {
        if (__builtin_popcount(d) % 2 == 0) {
            result(d) = ChiSpin(d >> 1);
        }
    }
    return result;
}



///// Utilities /////

// matrix power
template <typename MatrixType>
MatrixType matrixPower(const MatrixType& A, unsigned int n) {
    if (n == 0) {
        return MatrixType::Identity(A.rows(), A.cols());
    }
    if (n == 1) {
        return A;
    }
    MatrixType half = matrixPower(A, n / 2);
    if (n % 2 == 0) {
        return half * half;
    } else {
        return half * half * A;
    }
}





///// main /////

int main() {

    ////////////////////////
    ///// Initial settings
    ////////////////////////

    std::ifstream fin("cppPsiActionList.txt");
    for (int i = 0; i < 24; i++) {
        for (int d = 0; d < 4096; d++) {
            fin >> PsiActionList[i][d][0] >> PsiActionList[i][d][1];
        }
    }
    fin.close();

    fin.open("cppPsiPsiActionList.txt");
    for (int i = 0; i < 276; i++) {
        for (int d = 0; d < 2048; d++) {
            fin >> PsiPsiActionList[i][d][0] >> PsiPsiActionList[i][d][1];
        }
    }
    fin.close();


    
    ////////////////////////
    ///// Find the supercurrent as the common kernel of A-I and B-I
    ////////////////////////

    ///// Load files of the generators A and B

    std::vector<double> anglesA(12);
    fin.open("cppanglesA.txt");
    for (int i=0; i<12; i++) {
        fin >> anglesA.at(i);
    }
    fin.close();

    std::vector<double> anglesB(12);
    fin.open("cppanglesB.txt");
    for (int i=0; i<12; i++) {
        fin >> anglesB.at(i);
    }
    fin.close();

    Eigen::MatrixXd eigenmatA(24, 24);
    fin.open("cppeigenmatA.txt");
    for (int i = 0; i < 24; i++)
        for (int j = 0; j < 24; j++)
            fin >> eigenmatA(i, j);
    fin.close();

    Eigen::MatrixXd eigenmatB(24, 24);
    fin.open("cppeigenmatB.txt");
    for (int i = 0; i < 24; i++)
        for (int j = 0; j < 24; j++)
            fin >> eigenmatB(i, j);
    fin.close();

    ///// Initialize variables

    int n = 2048;
    std::ofstream fout;
    
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    auto now = clock::now();
    std::chrono::duration<double> elapsed = now - start;

    ////////////////////////
    // Step 1: Generate representation matrices of A and B on positive-chiral spinors
    ////////////////////////

    ///// Generate representation matrices of A and B on positive-chiral spinors

    /*
    // Generate representation matrix of A on positive-chiral spinors
    Eigen::MatrixXcd A(n, n);
    for (int i = 0; i < n; i++) {
        Eigen::VectorXcd basisChi = Eigen::VectorXcd::Zero(n);
        basisChi[i] = std::complex<double>(1.0, 0.0);
        Eigen::VectorXcd v = SO24LiftActionChi(anglesA, eigenmatA, basisChi);
        A.col(i) = v; // record as a column vector
        cout << i << " "; // log
    }
    // Save
    fout.open("datacpp_RepMatOnChiSpinA.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fout << A(i,j).real() << " " << A(i,j).imag() << "  ";
        }
        fout << "\n";
    }
    fout.close();

    // Generate representation matrix of B on positive-chiral spinors
    Eigen::MatrixXcd B(n, n);
    for (int i = 0; i < n; i++) {
        Eigen::VectorXcd basisChi = Eigen::VectorXcd::Zero(n);
        basisChi[i] = std::complex<double>(1.0, 0.0);
        Eigen::VectorXcd v = SO24LiftActionChi(anglesB, eigenmatB, basisChi);
        B.col(i) = v; // record as a column vector
        cout << i << " "; // log
    }
    // Save
    fout.open("datacpp_RepMatOnChiSpinB.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fout << B(i,j).real() << " " << B(i,j).imag() << "  ";
        }
        fout << "\n";
    }
    fout.close();
    */

    ///// Sanity check: check the relations of generators

    /*
    Eigen::MatrixXcd BBABA = B*B*A*B*A;
    Eigen::MatrixXcd BBABA3 = BBABA * BBABA * BBABA;
    cout << (BBABA3*BBABA3 - Eigen::MatrixXd::Identity(n,n)).norm() << endl;
    // Eigen::MatrixXcd ABABB = A*B*A*B*B;
    // Eigen::MatrixXcd ABABB3 = ABABB * ABABB * ABABB;
    // cout << (ABABB3*ABABB3 - Eigen::MatrixXd::Identity(n,n)).norm() << endl;
    Eigen::MatrixXcd BA = B*A;
    cout << (matrixPower(BA, 40) - Eigen::MatrixXd::Identity(n,n)).norm() << endl;
    // Eigen::MatrixXcd AB = A*B;
    // cout << (matrixPower(AB, 40) - Eigen::MatrixXd::Identity(n,n)).norm() << endl;
    Eigen::MatrixXcd longword = B*B*A*B*B*A*B*A*B*B*A*B*A*B*A*B*A;
    cout << (matrixPower(longword, 33) - Eigen::MatrixXd::Identity(n,n)).norm() << endl;
    // Eigen::MatrixXcd longwordrev = A*B*A*B*A*B*A*B*B*A*B*A*B*B*A*B*B;
    // cout << (matrixPower(longwordrev, 33) - Eigen::MatrixXd::Identity(n,n)).norm() << endl;
    */

    ///// Load the representation matrices of A and B

    /*
    Eigen::MatrixXcd A(n, n);
    fin.open("datacpp_RepMatOnChiSpinA.txt");
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            double re, im;
            fin >> re >> im;
            A(i,j) = complex<double>(re, im);
        }
    }
    fin.close();

    Eigen::MatrixXcd B(n, n);
    fin.open("datacpp_RepMatOnChiSpinB.txt");
    for (int i = 0; i < B.rows(); i++) {
        for (int j = 0; j < B.cols(); j++) {
            double re, im;
            fin >> re >> im;
            B(i,j) = complex<double>(re, im);
        }
    }
    fin.close();

    cout << "The representation matrices of A and B are loaded." << endl;
    */

    ////////////////////////
    // Step 2: Compute the common kernel of A-I and B-I
    ////////////////////////

    /*

    ///// Find a basis QA of Ker(A-I) and a basis QB of Ker(B-I)

    // A-I and B-I
    Eigen::MatrixXcd AI = A - Eigen::MatrixXcd::Identity(n,n);
    Eigen::MatrixXcd BI = B - Eigen::MatrixXcd::Identity(n,n);

    // The following code uses SVD to construct bases of kernels, but it crashes due to insufficient memory.
    // const double eps = std::numeric_limits<double>::epsilon();
    // Eigen::BDCSVD<Eigen::MatrixXcd> svdAI(AI, Eigen::ComputeThinV);
    // const auto sAI = svdAI.singularValues();
    // double tauAI = std::max(n,n) * eps * sAI(0);
    // int dimKerAI = (sAI.array() < tauAI).count();
    // Eigen::MatrixXcd QA = svdAI.matrixV().rightCols(dimKerAI); // basis of kernel of A-I
    // Eigen::BDCSVD<Eigen::MatrixXcd> svdBI(BI, Eigen::ComputeThinV);
    // const auto sBI = svdBI.singularValues();
    // double tauBI = std::max(n,n) * eps * sBI(0);
    // int dimKerBI = (sBI.array() < tauBI).count();
    // Eigen::MatrixXcd QB = svdBI.matrixV().rightCols(dimKerBI); // basis of kernel of B-I

    // So, we use LU decomposition to construct bases of kernels as follows.
    Eigen::FullPivLU<Eigen::MatrixXcd> luAI(AI);
    luAI.setThreshold(1e-12);
    Eigen::MatrixXcd QA = luAI.kernel(); // basis of kernel of A-I
    
    Eigen::FullPivLU<Eigen::MatrixXcd> luBI(BI);
    luBI.setThreshold(1e-12);
    Eigen::MatrixXcd QB = luBI.kernel(); // basis of kernel of B-I

    cout << "rank Ker(A-I) = " << QA.cols() << ", rank Ker(B-I) = " << QB.cols() << endl;

    // orthonormalize the bases of kernels
    auto orthonormalize = [&](const Eigen::MatrixXcd& X){
        Eigen::HouseholderQR<Eigen::MatrixXcd> qr(X);
        Eigen::MatrixXcd Q = Eigen::MatrixXcd::Identity(X.rows(), X.cols());
        Q = qr.householderQ() * Q;
        return Q;
    };
    QA = orthonormalize(QA);
    QB = orthonormalize(QB);

    // Sanity checks
    std::cout << "||QA^*QA - I|| = " << (QA.adjoint()*QA - Eigen::MatrixXcd::Identity(QA.cols(), QA.cols())).norm() << "\n";
    std::cout << "||QB^*QB - I|| = " << (QB.adjoint()*QB - Eigen::MatrixXcd::Identity(QB.cols(), QB.cols())).norm() << "\n";
    fout.open("test_precisionQAQB.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int j=0; j<QA.cols(); ++j) {
        fout << "A-res col " << j << " = " << (A*QA.col(j) - QA.col(j)).norm() << "\n";
    }
    for (int j=0; j<QB.cols(); ++j) {
        fout << "B-res col " << j << " = " << (B*QB.col(j) - QB.col(j)).norm() << "\n";
    }
    fout.close();

    ///// Solve QA * x = QB * y, by finding kernel of C = [QA, -QB]

    const int p = QA.cols();
    const int q = QB.cols();
    Eigen::MatrixXcd C(n, p + q); // C = [QA, -QB] (size: n x (p+q))
    C.leftCols(p) = QA;
    C.rightCols(q) = -QB;

    // SVD of C = [QA, -QB] = U S V*
    // Just in the case of p+q > n, it is safe to require FullV.
    Eigen::JacobiSVD<Eigen::MatrixXcd> svdC(C, Eigen::ComputeThinU | Eigen::ComputeFullV);

    // Sanity check
    const Eigen::VectorXd sC = svdC.singularValues();
    fout.open("test_SingularValuesQA-QB.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    fout << "size: " << sC.size() << endl;
    for (int i=0; i<sC.size(); i++) fout << sC[i] << "\n"; // The last singular value must be (numerically) zero.
    fout.close();

    // z = [x^T, y^T]^T
    const Eigen::VectorXcd z = svdC.matrixV().col(svdC.matrixV().cols() - 1);
    Eigen::VectorXcd x = z.topRows(p);
    Eigen::VectorXcd y = z.bottomRows(q);

    ///// The supercurrent is u = QA * x = QB * y

    Eigen::VectorXcd uA = QA * x;
    Eigen::VectorXcd uB = QB * y;
    Eigen::VectorXcd u = uA +uB; // take average to stabilize u numerically

    // Normalize u
    if (uA.norm() > 0) uA /= uA.norm();
    if (uB.norm() > 0) uB /= uB.norm();
    if (u.norm() > 0) u /= u.norm();

    // Sanity checks; everything must be numerically zero
    cout << "||C z|| = " << (C * z).norm() << "\n";
    cout << "||QA x - QB y|| = " << (uA - uB).norm() << "\n";
    cout << "||A uA - uA|| = " << (A*uA - uA).norm() << "\n";
    cout << "||A u - u|| = " << (A*u - uA).norm() << "\n";
    cout << "||B uB - uB|| = " << (B*uB - uB).norm() << "\n";
    cout << "||B u - u|| = " << (B*u - u).norm() << "\n";

    // Save the supercurrent (uA, uB, and) u

    // fout.open("datacpp_supercurrentA_ChiSpin.txt");
    // fout.setf(std::ios::scientific);
    // fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    // for (int i = 0; i < uA.size(); i++) { fout << uA(i).real() << " " << uA(i).imag() << endl; }
    // fout.close();

    // fout.open("datacpp_supercurrentB_ChiSpin.txt");
    // fout.setf(std::ios::scientific);
    // fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    // for (int i = 0; i < uB.size(); i++) { fout << uB(i).real() << " " << uB(i).imag() << endl; }
    // fout.close();

    fout.open("datacpp_supercurrent_ChiSpin.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < u.size(); i++) { fout << u(i).real() << " " << u(i).imag() << endl; }
    fout.close();

    */

    ////////////////////////
    // Step 3: Sanity checks
    ////////////////////////

    ///// Load the supercurrent

    Eigen::VectorXcd sc(n);
    fin.open("datacpp_supercurrent_ChiSpin.txt");
    for (int i = 0; i < n; i++) {
        double re, im;
        fin >> re >> im;
        sc(i) = complex<double>(re, im);
    }
    fin.close();
    cout << "The supercurrent (as a positive-chiral spinor) is loaded." << endl;

    // Make positive-chiral spinor into full spinor
    sc = ChiSpinToFullSpin(sc);

    ///// Sanity checks of the supercurrent
    
    cout << "Sanity Checks of the supercurrent" << endl;

    // invariant under the generators A and B
    Eigen::VectorXcd A_sc = SO24LiftAction(anglesA, eigenmatA, sc);
    cout << (sc - A_sc).norm() << endl; // must be 0
    cout << (sc + A_sc).norm() << endl; // must be 2
    Eigen::VectorXcd B_sc = SO24LiftAction(anglesB, eigenmatB, sc);
    cout << (sc - B_sc).norm() << endl; // must be 0
    cout << (sc + B_sc).norm() << endl; // must be 2

    // invariant under the conjugacy class of commuting pair (8,44)
    vector<double> angles_8_44_first(24), angles_8_44_second(24);
    fin.open("cpp8_44_anglesfirst.txt");
    for (int i=0; i<24; i++) {
        fin >> angles_8_44_first.at(i);
    }
    fin.close();
    fin.open("cpp8_44_anglessecond.txt");
    for (int i=0; i<24; i++) {
        fin >> angles_8_44_second.at(i);
    }
    fin.close();

    Eigen::MatrixXd eigenmat_8_44(24, 24);
    fin.open("cpp8_44_eigenmat.txt");
    for (int i = 0; i < 24; i++)
        for (int j = 0; j < 24; j++)
            fin >> eigenmat_8_44(i, j);
    fin.close();

    Eigen::VectorXcd M_8_44_first_sc = SO24LiftAction(angles_8_44_first, eigenmat_8_44, sc);
    // One of the followings must be 0, and the other 2.
    cout << (sc - M_8_44_first_sc).norm() << endl;
    cout << (sc + M_8_44_first_sc).norm() << endl;
    Eigen::VectorXcd M_8_44_second_sc = SO24LiftAction(angles_8_44_second, eigenmat_8_44, sc);
    // One of the followings must be 0, and the other 2.
    cout << (sc - M_8_44_second_sc).norm() << endl;
    cout << (sc + M_8_44_second_sc).norm() << endl;

    

    ////////////////////////
    ///// The followings are unnecessary
    ////////////////////////

    ///// Compute eigenvectors of A and B

    /*
    // Compute eigenvectors of A
    // A is not Hermitian, so we cannot use SelfAdjointEigenSolver.
    start = clock::now();

    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> Aces(A); // computation is done here
    Eigen::VectorXcd  Aevals = Aces.eigenvalues();
    Eigen::MatrixXcd  Aevecs = Aces.eigenvectors();
    
    now = clock::now();
    elapsed = now - start;
    cout << "EigenSolver time: " << elapsed.count() << endl;

    // Save
    fout.open("Aevals.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < Aevals.size(); i++) {
        fout << Aevals(i).real() << " " << Aevals(i).imag() << "\n";
    }
    fout.close();   

    fout.open("Aevecs.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < Aevecs.rows(); i++) {
        for (int j = 0; j < Aevecs.cols(); j++) {
            fout << Aevecs(i,j).real() << " " << Aevecs(i,j).imag() << "  ";
        }
        fout << "\n";
    }
    fout.close();

    // Compute eigenvectors of B
    start = clock::now();

    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> Bces(B); // computation is done here
    Eigen::VectorXcd  Bevals = Bces.eigenvalues();
    Eigen::MatrixXcd  Bevecs = Bces.eigenvectors();
    
    now = clock::now();
    elapsed = now - start;
    cout << "EigenSolver time: " << elapsed.count() << endl;

    // Save
    fout.open("Bevals.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < Bevals.size(); i++) {
        fout << Bevals(i).real() << " " << Bevals(i).imag() << "\n";
    }
    fout.close();

    fout.open("Bevecs.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < Bevecs.rows(); i++) {
        for (int j = 0; j < Bevecs.cols(); j++) {
            fout << Bevecs(i,j).real() << " " << Bevecs(i,j).imag() << "  ";
        }
        fout << "\n";
    }
    fout.close();
    */

    ///// Load the eigenvalues and eigenvectors

    /*
    Eigen::VectorXcd  Aevals(n);
    Eigen::MatrixXcd  Aevecs(n, n);
    fin.open("Aevals.txt");
    // fin >> std::hexfloat;
    for (int i = 0; i < n; i++) {
        double re, im;
        fin >> re >> im;
        Aevals(i) = complex<double>(re, im);
    }
    fin.close();

    fin.open("Aevecs.txt");
    // fin >> std::hexfloat;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double re, im;
            fin >> re >> im;
            Aevecs(i,j) = complex<double>(re, im);
        }
    }
    fin.close();

    Eigen::VectorXcd  Bevals(n);
    Eigen::MatrixXcd  Bevecs(n, n);
    fin.open("Bevals.txt");
    // fin >> std::hexfloat;
    for (int i = 0; i < n; i++) {
        double re, im;
        fin >> re >> im;
        Bevals(i) = complex<double>(re, im);
    }
    fin.close();

    fin.open("Bevecs.txt");
    // fin >> std::hexfloat;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double re, im;
            fin >> re >> im;
            Bevecs(i,j) = complex<double>(re, im);
        }
    }
    fin.close();
    */

    ///// Extract invariant subspaces

    /*
    vector<Eigen::VectorXcd>  AinvEvecsVec;
    for (int i=0; i<n; i++) {
        if (Aevals(i).real() > 0) { // eval of A is 1 or -1.
            AinvEvecsVec.push_back(Aevecs.col(i));
        }
    }
    Eigen::MatrixXcd AinvEvecs(n, static_cast<int>(AinvEvecsVec.size()));
    for (int j = 0; j < AinvEvecs.cols(); j++) AinvEvecs.col(j) = AinvEvecsVec[j];

    vector<Eigen::VectorXcd>  BinvEvecsVec;
    for (int i=0; i<n; i++) {
        if (Bevals(i).real() > 0) { // eval of B is 1 or exp(i2pi/3) or exp(i4pi/3).
            BinvEvecsVec.push_back(Bevecs.col(i));
        }
    }
    Eigen::MatrixXcd BinvEvecs(n, static_cast<int>(BinvEvecsVec.size()));
    for (int j = 0; j < BinvEvecs.cols(); j++) BinvEvecs.col(j) = BinvEvecsVec[j];

    cout << "invariant eigenvectors are extracted." << endl;
    cout << "A: " << AinvEvecs.cols() << " invariant vectors." << endl;
    cout << "B: " << BinvEvecs.cols() << " invariant vectors." << endl;

    // Save
    fout.open("AinvEvecs.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < AinvEvecs.rows(); i++) {
        for (int j = 0; j < AinvEvecs.cols(); j++) {
            fout << AinvEvecs(i,j).real() << " " << AinvEvecs(i,j).imag() << "  ";
        }
        fout << "\n";
    }
    fout.close();

    fout.open("BinvEvecs.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < BinvEvecs.rows(); i++) {
        for (int j = 0; j < BinvEvecs.cols(); j++) {
            fout << BinvEvecs(i,j).real() << " " << BinvEvecs(i,j).imag() << "  ";
        }
        fout << "\n";
    }
    fout.close();
    */

    ///// Sanity Checks
    
    /*
    Eigen::VectorXcd B_ev0 = SO24LiftActionChi(anglesB, eigenmatB, BinvEvecs.col(0));
    cout << "Sanity Check of BinvEvecs.col(0): " << (BinvEvecs.col(0) - B_ev0).norm() << endl;
    cout << "Sanity Check of BinvEvecs.col(0): " << (BinvEvecs.col(0) + B_ev0).norm() << endl;

    fout.open("test_precision_invsubsp.txt");
    fout.setf(std::ios::scientific);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int j=0; j<AinvEvecs.cols(); ++j) {
        fout << "A-res col " << j << " = " << (A*AinvEvecs.col(j) - AinvEvecs.col(j)).norm() << "\n";
    }
    for (int j=0; j<BinvEvecs.cols(); ++j) {
        fout << "B-res col " << j << " = " << (B*BinvEvecs.col(j) - BinvEvecs.col(j)).norm() << "\n";
    }
    fout.close();
    */

    ///// Load invariant subspaces

    /*
    Eigen::MatrixXcd AinvEvecs(n, 1056);
    Eigen::MatrixXcd BinvEvecs(n, 680);

    fin.open("AinvEvecs.txt");
    for (int i = 0; i < AinvEvecs.rows(); i++) {
        for (int j = 0; j < AinvEvecs.cols(); j++) {
            double re, im;
            fin >> re >> im;
            AinvEvecs(i,j) = complex<double>(re, im);
        }
    }

    fin.open("BinvEvecs.txt");
    for (int i = 0; i < BinvEvecs.rows(); i++) {
        for (int j = 0; j < BinvEvecs.cols(); j++) {
            double re, im;
            fin >> re >> im;
            BinvEvecs(i,j) = complex<double>(re, im);
        }
    }
    fin.close();
    */
}