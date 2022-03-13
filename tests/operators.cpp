#include <iostream>
#include "../fsgrid.hpp"
/*
  Copyright (C) 2016 Finnish Meteorological Institute
  Copyright (C) 2016 CSC -IT Center for Science 

  This file is part of fsgrid
 
  fsgrid is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  fsgrid is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY;
  without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with fsgrid.  If not, see <http://www.gnu.org/licenses/>.
*/



int main(int argc, char** argv) {
   typedef float Real;
   MPI_Init(&argc,&argv);

   int rank,size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   // Create a 8×8 Testgrid
   std::array<int32_t, 3> globalSize{2000,1000,1};
   std::array<int32_t, 3> globalSize2{3000,1000,1};
   std::array<bool, 3> isPeriodic{false,false,true};



   FsGridCouplingInformation gridCoupling;
   FsGrid<std::array<Real, 1>, 2> A(globalSize, MPI_COMM_WORLD, isPeriodic, gridCoupling);
   FsGrid<std::array<Real, 1>, 2> B(globalSize, MPI_COMM_WORLD, isPeriodic, gridCoupling);
   FsGrid<std::array<Real, 1>, 2> C(globalSize, MPI_COMM_WORLD, isPeriodic, gridCoupling);
   FsGrid<std::array<Real, 1>, 2> D(globalSize2, MPI_COMM_WORLD, isPeriodic, gridCoupling);

  //  FsGrid<std::array<float, 1>, 2> newGuy(A + B);
   //  + 
   A=B+C;
  //  +=
  A+=B;
  // -
  A=A-C;
  // -=
  A-=B;
  D=D*3.0;
  C=lerp_t(A,B,5.,10.,7.);
  //supposed to fail;
  // A=B+D;



















   MPI_Finalize();
   return 0;
}
