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
    ///// Calibrate the angles for each representative of SL(2,Z)-orbit
    ////////////////////////

    ///// Initialize variables

    int n = 2048;
    std::ofstream fout, flog;
    
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    auto now = clock::now();
    std::chrono::duration<double> elapsed = now - start;



    ///// Load the supercurrent

    Eigen::VectorXcd sc(n);
    fin.open("datacpp_supercurrent_ChiSpin.txt");
    for (int i = 0; i < n; i++) {
        double re, im;
        fin >> re >> im;
        sc(i) = complex<double>(re, im);
    }
    fin.close();

    // Make positive-chiral spinor into full spinor
    sc = ChiSpinToFullSpin(sc);

    cout << "The supercurrent is prepared." << endl;



    ///// Calibrate the angles for each representative of SL(2,Z)-orbit

    int num_reps = 348; // number of representatives
    double eps = 1e-11;

    vector<int> ind_of_rep(2);
    vector<double> angles_first(12), angles_second(12);
    Eigen::MatrixXd eigenmat(24, 24);

    fin.open("cppAngleAndEigenmat.txt");
    fout.open("dataIsCalibrationNeeded.m");
    fout << "IsCalibrationNeeded = <|";
    flog.open("log_ReprActionOnSupercurrent.txt");
    
    for (int r = 0; r < num_reps; r++) {
        cout << r << " "; // log
        if (r > 0) {
            fout << "," << "\n";
        }

        // Load data
        fin >> ind_of_rep.at(0) >> ind_of_rep.at(1);
        for (int i = 0; i < 12; i++)
            fin >> angles_first.at(i);
        for (int i = 0; i < 12; i++)
            fin >> angles_second.at(i);
        for (int i = 0; i < 24; i++)
            for (int j = 0; j < 24; j++)
                fin >> eigenmat(i, j);

        // Calculate whether the supercurrent is preserved or reversed
        Eigen::VectorXcd first_sc = SO24LiftAction(angles_first, eigenmat, sc);
        // One of the followings must be 0, and the other 2.
        double first_preserved = (sc - first_sc).norm();
        double first_reversed = (sc + first_sc).norm();
        Eigen::VectorXcd second_sc = SO24LiftAction(angles_second, eigenmat, sc);
        // One of the followings must be 0, and the other 2.
        double second_preserved = (sc - second_sc).norm();
        double second_reversed = (sc + second_sc).norm();

        flog << "(" << ind_of_rep.at(0) << ", " << ind_of_rep.at(1) << ")" << "\n";
        flog << first_preserved << " " << first_reversed << " " << second_preserved << " " << second_reversed << "\n";

        // Judge whether the angles need to be calibrated or not
        fout << "{" << ind_of_rep.at(0) << ", " << ind_of_rep.at(1) << "} -> {";
        // Deal with the first set of angles
        if (first_preserved < eps && abs(2 - first_reversed) < eps) {
            fout << 0;
        }
        else if (abs(2 - first_preserved) < eps && first_reversed < eps)
        {
            fout << 1;
        }
        else {
            cout << "Something is wrong at the first set of angles of (" << ind_of_rep.at(0) << ", " << ind_of_rep.at(1) << ")" << "\n";
            cout << first_preserved << " " << abs(2 - first_reversed) << " " << abs(2 - first_preserved) << " " << first_reversed << "\n";
            break;
        }
        fout << ", ";
        // Deal with the second set of angles
        if (second_preserved < eps && abs(2 - second_reversed) < eps) {
            fout << 0;
        }
        else if (abs(2 - second_preserved) < eps && second_reversed < eps)
        {
            fout << 1;
        }
        else {
            cout << "Something is wrong at the second set of angles of (" << ind_of_rep.at(0) << ", " << ind_of_rep.at(1) << ")" << "\n";
            cout << second_preserved << " " << abs(2 - second_reversed) << " " << abs(2 - second_preserved) << " " << second_reversed << "\n";
            break;
        }
        fout << "}";
    }

    flog.close();
    fout << "|>";
    fout.close();
    fin.close();
}