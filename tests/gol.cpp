/*
Conway's Game of Life test for fsgrid.

Copyright 2017 Ilja Honkonen

This file is part of fsgrid

fsgrid is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

fsgrid is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with fsgrid.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <array>
#include <iostream>
#include "../fsgrid.hpp"

int main(int argc, char** argv) try {
   MPI_Init(&argc,&argv);

   int rank,size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   FsGrid<std::array<int, 2>, 1> grid({10, 10, 1}, MPI_COMM_WORLD, {true, true, false});
   const std::array<int, 3> localSize = grid.getLocalSize();

   for (int z = 0; z < localSize[2]; z++) {
   for (int y = 0; y < localSize[1]; y++) {
   for (int x = 0; x < localSize[0]; x++) {
      auto* const cell = grid.get(x, y, z);
      if (cell == nullptr) {
         std::cout << __FILE__ "(" << __LINE__ << ")" << std::endl;
         abort();
	  }

      *cell = {0, 0};
      // create glider in upper left
      const auto index = grid.getGlobalIndices(x, y, z);
      if (
         (index[0] == 1 and index[1] == 2)
         or (index[0] == 2 and index[1] == 3)
         or (index[0] == 3 and index[1] == 3)
         or (index[0] == 3 and index[1] == 2)
         or (index[0] == 3 and index[1] == 1)
      ) {
         (*cell)[0] = 1;
	  }
   }}}

   for (size_t turn = 0; turn < 10; turn++) {
	  grid.updateGhostCells();

      // get number of live neighbors
      for (int z = 0; z < localSize[2]; z++) {
      for (int y = 0; y < localSize[1]; y++) {
      for (int x = 0; x < localSize[0]; x++) {
         auto* const cell = grid.get(x, y, z);
         if (cell == nullptr) {
            std::cout << __FILE__ "(" << __LINE__ << ")" << std::endl;
            abort();
         }
         for (auto x_offset: {-1, 0, +1}) {
         for (auto y_offset: {-1, 0, +1}) {
         for (auto z_offset: {0}) {
            if (x_offset == 0 and y_offset == 0 and z_offset == 0) {
               continue;
			}

            const auto* const neighbor = grid.get(x + x_offset, y + y_offset, z + z_offset);
            if (neighbor == nullptr) {
               std::cout << __FILE__ "(" << __LINE__ << ")" << std::endl;
               abort();
	        }
            if ((*neighbor)[0] > 0) {
               (*cell)[1]++;
			}
		 }}}
	  }}}

      // set new state
      for (int z = 0; z < localSize[2]; z++) {
      for (int y = 0; y < localSize[1]; y++) {
      for (int x = 0; x < localSize[0]; x++) {
         auto* const cell = grid.get(x, y, z);
         if (cell == nullptr) {
            std::cout << __FILE__ "(" << __LINE__ << ")" << std::endl;
            abort();
         }
         if ((*cell)[1] == 3) {
            (*cell)[0] = 1;
		 } else if ((*cell)[1] != 2) {
            (*cell)[0] = 0;
		 }
         (*cell)[1] = 0;
	  }}}

      // check result
      int local_alive = 0, global_alive = 0;
      for (int z = 0; z < localSize[2]; z++) {
      for (int y = 0; y < localSize[1]; y++) {
      for (int x = 0; x < localSize[0]; x++) {
         const auto* const cell = grid.get(x, y, z);
         if (cell == nullptr) {
            std::cout << __FILE__ "(" << __LINE__ << ")" << std::endl;
            abort();
         }
         if ((*cell)[0] > 0) {
            local_alive++;
         }
	  }}}
      MPI_Reduce(&local_alive, &global_alive, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rank == 0 and global_alive != 5) {
         std::cout << global_alive << std::endl;
         abort();
      }
   }

   MPI_Finalize();
   return EXIT_SUCCESS;

} catch(...) {

   std::cout << "Failed!" << std::endl;
   return EXIT_FAILURE;
}
