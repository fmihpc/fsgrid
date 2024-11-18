#pragma once

/*
  Copyright (C) 2016 Finnish Meteorological Institute
  Copyright (C) 2016-2024 CSC -IT Center for Science

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
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <stdint.h>

namespace FsGridTools {
// Size type for global array indices
typedef uint32_t FsSize_t;
// Size type for global/local array indices, incl. possible negative values
typedef int32_t FsIndex_t;
typedef int64_t LocalID;
typedef int64_t GlobalID;
typedef int32_t Task_t;

//! Helper function: calculate size of the local coordinate space for the given dimension
// \param numCells Number of cells in the global Simulation, in this dimension
// \param numTasks Total number of tasks in this dimension
// \param taskIndex This task's position in this dimension
// \return Number of cells for this task's local domain (actual cells, not counting ghost cells)
constexpr static FsIndex_t calcLocalSize(FsSize_t numCells, Task_t numTasks, Task_t taskIndex) {
   const FsIndex_t nPerTask = static_cast<Task_t>(numCells) / numTasks;
   const FsIndex_t remainder = static_cast<Task_t>(numCells) % numTasks;
   return nPerTask + (taskIndex < remainder);
}

//! Helper function: calculate position of the local coordinate space for the given dimension
// \param numCells number of cells
// \param numTasks Total number of tasks in this dimension
// \param taskIndex This task's position in this dimension
// \return Cell number at which this task's domains cells start (actual cells, not counting ghost cells)
constexpr static FsIndex_t calcLocalStart(FsSize_t numCells, Task_t numTasks, Task_t taskIndex) {
   const FsIndex_t remainder = static_cast<Task_t>(numCells) % numTasks;
   return taskIndex * calcLocalSize(numCells, numTasks, taskIndex) + (taskIndex >= remainder) * remainder;
}

//! Helper function: given a global cellID, calculate the global cell coordinate from it.
// This is then used do determine the task responsible for this cell, and the
// local cell index in it.
constexpr static std::array<FsIndex_t, 3> globalIDtoCellCoord(GlobalID id, const std::array<FsSize_t, 3>& globalSize) {
   // We're returning FsIndex_t, which is int32_t. globalSizes are FsSize_t, which are uint32_t.
   // Need to check that the globalSizes aren't larger than maximum of int32_t
   const bool globalSizeOverflows = globalSize[0] > std::numeric_limits<FsIndex_t>::max() ||
                                    globalSize[1] > std::numeric_limits<FsIndex_t>::max() ||
                                    globalSize[2] > std::numeric_limits<FsIndex_t>::max();
   const bool idNegative = id < 0;

   // To avoid overflow, do this instead of checking for id < product of globalSize
   // The number of divisions stays the same anyway
   const GlobalID idPerGs0 = id / globalSize[0];
   const GlobalID idPerGs0PerGs1 = idPerGs0 / globalSize[1];
   const bool idTooLarge = idPerGs0PerGs1 >= globalSize[2];

   const bool badInput = idTooLarge || idNegative || globalSizeOverflows;

   if (badInput) {
      // For bad input, return bad output
      return {
          std::numeric_limits<FsIndex_t>::min(),
          std::numeric_limits<FsIndex_t>::min(),
          std::numeric_limits<FsIndex_t>::min(),
      };
   } else {
      return {
          (FsIndex_t)(id % globalSize[0]),
          (FsIndex_t)(idPerGs0 % globalSize[1]),
          (FsIndex_t)(idPerGs0PerGs1 % globalSize[2]),
      };
   }
}

//! Helper function to optimize decomposition of this grid over the given number of tasks
constexpr static std::array<Task_t, 3> computeDomainDecomposition(const std::array<FsSize_t, 3>& globalSize,
                                                                  Task_t nProcs, int32_t numGhostCells = 1) {
   const std::array minDomainSize = {
       globalSize[0] == 1 ? 1 : numGhostCells,
       globalSize[1] == 1 ? 1 : numGhostCells,
       globalSize[2] == 1 ? 1 : numGhostCells,
   };
   const std::array maxDomainSize = {
       std::min(nProcs, static_cast<Task_t>(globalSize[0] / static_cast<FsSize_t>(minDomainSize[0]))),
       std::min(nProcs, static_cast<Task_t>(globalSize[1] / static_cast<FsSize_t>(minDomainSize[1]))),
       std::min(nProcs, static_cast<Task_t>(globalSize[2] / static_cast<FsSize_t>(minDomainSize[2]))),
   };

   int64_t minimumCost = std::numeric_limits<int64_t>::max();
   std::array dd = {1, 1, 1};
   for (Task_t i = 1; i <= maxDomainSize[0]; i++) {
      for (Task_t j = 1; j <= maxDomainSize[1]; j++) {
         const Task_t k = nProcs / (i * j);
         if (k == 0) {
            break;
         }

         // No need to optimize an incompatible DD, also checks for missing remainders
         if (i * j * k != nProcs || k > maxDomainSize[2]) {
            continue;
         }

         const std::array processBox = {
             calcLocalSize(globalSize[0], i, 0),
             calcLocalSize(globalSize[1], j, 0),
             calcLocalSize(globalSize[2], k, 0),
         };

         // clang-format off
         const int64_t baseCost = (i > 1 ? processBox[1] * processBox[2] : 0)
                                + (j > 1 ? processBox[0] * processBox[2] : 0)
                                + (k > 1 ? processBox[0] * processBox[1] : 0);
         const int64_t neighborMultiplier = (i != 1 && j != 1 && k != 1) * 13
                                          + (i == 1 && j != 1 && k != 1) * 4
                                          + (i != 1 && j == 1 && k != 1) * 4
                                          + (i != 1 && j != 1 && k == 1) * 4;
         // clang-format on
         const int64_t cost = baseCost * neighborMultiplier;
         if (cost < minimumCost) {
            minimumCost = cost;
            dd = {i, j, k};
         }
      }
   }

   if (minimumCost == std::numeric_limits<int64_t>::max() || (Task_t)(dd[0] * dd[1] * dd[2]) != nProcs) {
      throw std::runtime_error("FSGrid computeDomainDecomposition failed");
   }

   return dd;
}

template <typename... Args> void cerrArgs(Args...);
template <> inline void cerrArgs() {}
template <typename T> void cerrArgs(T t) { std::cerr << t << "\n"; }
template <typename Head, typename... Tail> void cerrArgs(Head head, Tail... tail) {
   cerrArgs(head);
   cerrArgs(tail...);
}

// Recursively write all arguments to cerr if status is not success, then throw
template <typename... Args> void writeToCerrAndThrowIfFailed(bool failed, Args... args) {
   if (failed) {
      cerrArgs(args...);
      throw std::runtime_error("Unrecoverable error encountered in FsGrid, consult cerr for more information");
   }
}

template <typename... Args> void mpiCheck(int status, Args... args) {
   writeToCerrAndThrowIfFailed(status != MPI_SUCCESS, args...);
}

template <typename... Args> void debugAssert([[maybe_unused]] bool condition, [[maybe_unused]] Args... args) {
#ifdef FSGRID_DEBUG
   writeToCerrAndThrowIfFailed(condition, args...);
#endif
}
} // namespace FsGridTools
