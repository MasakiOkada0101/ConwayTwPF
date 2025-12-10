20251210

1. About C++ codes
C++ codes are only used to generate data needed for the computation of the twisted partition functions.
More precisely,
- "FindSupercurrent.cpp" finds the supercurrent of Duncan's module.
- "CalibrateAngles.cpp" calibrates the angles of each commuting pair of the Conway group, using the supercurrent.
Therefore, basically, there is no need to rerun them.
But if you rerun them, here are some caveats:
- written with the library "Eigen" version 3.4.0.
- written with C++17. If you use C++20, you can replace __builtin_popcount() with std::popcount().
- consider to use -O3 optimization of GCC. It reduces the computational time dramatically. Otherwise, some calculations might take too much time. For example, the SVD in "FindSupercurrent.cpp" took more than 10 minutes even under the -O3 optimization.
- "datacpp_RepMatOnChiSpinA.txt" and "datacpp_RepMatOnChiSpinB.txt" are too big (about 200MB) to upload to GitHub. So please generate them by using "FindSupercurrent.cpp" on your computer.


2. About Mathematica codes
"SL2ZOrbits.nb" generates "dataSL2ZOrbits.m" from "gl2z-action.m".
"PF.nb" generates files to pass to C++ code, and compute the twisted partition functions.


3. About filenames
"data...m" : created by Mathematica code, and used in Mathematica code.
"cpp...txt" : created by Mathematica code, and used in C++ code.
"datacpp...txt" : created by C++ code, and used in C++ code.

exceptions:
"data.m" : created by Mathematica and GAP in advance, and used in Mathematica code "PF.nb".
"gl2z-action.m" : created by Mathematica and GAP in advance, and used in Mathematica code "SL2ZOrbits.nb".
"dataIsCalibrationNeeded.m" : created by C++ code "CalibrateAngles.cpp", and used in Mathematica code "PF.nb".