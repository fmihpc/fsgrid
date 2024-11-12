#include "grid.hpp"
#include <gtest/gtest.h>
#include <mpi.h>
#include <tools.hpp>

TEST(FsGridTest, localToGlobalRoundtrip1) {
   const std::array<FsGridTools::FsSize_t, 3> globalSize{1024, 666, 71};
   const MPI_Comm parentComm = MPI_COMM_WORLD;
   const std::array<bool, 3> periodic{true, true, false};

   const auto grid =
       FsGrid<std::array<double, 15>, 1>(globalSize, parentComm, periodic, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
   const auto localSize = grid.getLocalSize();
   for (int32_t x = 0; x < localSize[0]; x++) {
      for (int32_t y = 0; y < localSize[1]; y++) {
         for (int32_t z = 0; z < localSize[2]; z++) {
            const auto global = grid.localToGlobal(x, y, z);
            const auto local = grid.globalToLocal(global[0], global[1], global[2]);
            ASSERT_EQ(local[0], x);
            ASSERT_EQ(local[1], y);
            ASSERT_EQ(local[2], z);
         }
      }
   }
}

TEST(FsGridTest, myGlobalIDCorrespondsToMyTask) {
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   const std::array<FsGridTools::FsSize_t, 3> globalSize{6547, 16, 77};
   const MPI_Comm parentComm = MPI_COMM_WORLD;
   const std::array<bool, 3> periodic{true, false, false};

   const auto grid =
       FsGrid<std::array<double, 6>, 1>(globalSize, parentComm, periodic, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
   const auto localSize = grid.getLocalSize();
   for (int32_t x = 0; x < localSize[0]; x++) {
      for (int32_t y = 0; y < localSize[1]; y++) {
         for (int32_t z = 0; z < localSize[2]; z++) {
            const auto gid = grid.globalIDFromLocalCoordinates(x, y, z);
            const auto task = grid.getTaskForGlobalID(gid);
            ASSERT_EQ(task, rank);
            ASSERT_EQ(task, rank);
            ASSERT_EQ(task, rank);
         }
      }
   }
}

TEST(FsGridTest, localIdInBounds) {
   const std::array<FsGridTools::FsSize_t, 3> globalSize{647, 1, 666};
   const MPI_Comm parentComm = MPI_COMM_WORLD;
   const std::array<bool, 3> periodic{true, false, true};

   const auto grid =
       FsGrid<std::array<double, 50>, 1>(globalSize, parentComm, periodic, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
   const auto localSize = grid.getLocalSize();
   for (int32_t x = 0; x < localSize[0]; x++) {
      for (int32_t y = 0; y < localSize[1]; y++) {
         for (int32_t z = 0; z < localSize[2]; z++) {
            const auto lid = grid.localIDFromLocalCoordinates(x, y, z);
            ASSERT_TRUE(grid.localIdInBounds(lid));
         }
      }
   }
}

TEST(FsGridTest, getNonPeriodic) {
   const std::array<FsGridTools::FsSize_t, 3> globalSize{12, 6, 2048};
   const MPI_Comm parentComm = MPI_COMM_WORLD;
   const std::array<bool, 3> periodic{false, false, false};
   constexpr int32_t numGhostCells = 1;

   auto grid =
       FsGrid<std::array<double, 8>, numGhostCells>(globalSize, parentComm, periodic, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
   const auto localSize = grid.getLocalSize();
   for (int32_t x = 0; x < localSize[0]; x++) {
      for (int32_t y = 0; y < localSize[1]; y++) {
         for (int32_t z = 0; z < localSize[2]; z++) {
            ASSERT_NE(grid.get(x, y, z), nullptr);
         }
      }
   }

   ASSERT_EQ(grid.get(-numGhostCells, 0, 0), nullptr);
   ASSERT_EQ(grid.get(-numGhostCells - 1, 0, 0), nullptr);
   ASSERT_EQ(grid.get(grid.getLocalSize()[0] + numGhostCells, 0, 0), nullptr);
   ASSERT_EQ(grid.get(grid.getLocalSize()[0] + numGhostCells - 1, 0, 0), nullptr);

   ASSERT_EQ(grid.get(0, -numGhostCells, 0), nullptr);
   ASSERT_EQ(grid.get(0, -numGhostCells - 1, 0), nullptr);
   ASSERT_EQ(grid.get(0, grid.getLocalSize()[1] + numGhostCells, 0), nullptr);
   ASSERT_EQ(grid.get(0, grid.getLocalSize()[1] + numGhostCells - 1, 0), nullptr);

   // This depends on the position on the grid
   if (grid.getLocalStart()[2] == 0) {
      ASSERT_EQ(grid.get(0, 0, -numGhostCells), nullptr);
   } else {
      ASSERT_NE(grid.get(0, 0, -numGhostCells), nullptr);
   }

   ASSERT_EQ(grid.get(0, 0, -numGhostCells - 1), nullptr);

   // This depends on the position on the grid
   if (grid.getLocalStart()[2] + grid.getLocalSize()[2] == static_cast<FsGridTools::FsIndex_t>(globalSize[2])) {
      ASSERT_EQ(grid.get(0, 0, grid.getLocalSize()[2] + numGhostCells - 1), nullptr);
   } else {
      ASSERT_NE(grid.get(0, 0, grid.getLocalSize()[2] + numGhostCells - 1), nullptr);
   }
   ASSERT_EQ(grid.get(0, 0, grid.getLocalSize()[2] + numGhostCells), nullptr);
}

TEST(FsGridTest, getPeriodic) {
   const std::array<FsGridTools::FsSize_t, 3> globalSize{120, 5, 1048};
   const MPI_Comm parentComm = MPI_COMM_WORLD;
   const std::array<bool, 3> periodic{true, true, true};
   constexpr int32_t numGhostCells = 2;

   auto grid =
       FsGrid<std::array<double, 8>, numGhostCells>(globalSize, parentComm, periodic, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
   const auto localSize = grid.getLocalSize();
   for (int32_t x = 0; x < localSize[0]; x++) {
      for (int32_t y = 0; y < localSize[1]; y++) {
         for (int32_t z = 0; z < localSize[2]; z++) {
            ASSERT_NE(grid.get(x, y, z), nullptr);
         }
      }
   }

   ASSERT_NE(grid.get(-numGhostCells, 0, 0), nullptr);
   ASSERT_EQ(grid.get(-numGhostCells - 1, 0, 0), nullptr);
   ASSERT_EQ(grid.get(grid.getLocalSize()[0] + numGhostCells, 0, 0), nullptr);
   ASSERT_NE(grid.get(grid.getLocalSize()[0] + numGhostCells - 1, 0, 0), nullptr);

   ASSERT_NE(grid.get(0, -numGhostCells, 0), nullptr);
   ASSERT_EQ(grid.get(0, -numGhostCells - 1, 0), nullptr);
   ASSERT_EQ(grid.get(0, grid.getLocalSize()[1] + numGhostCells, 0), nullptr);
   ASSERT_NE(grid.get(0, grid.getLocalSize()[1] + numGhostCells - 1, 0), nullptr);

   ASSERT_NE(grid.get(0, 0, -numGhostCells), nullptr);
   ASSERT_EQ(grid.get(0, 0, -numGhostCells - 1), nullptr);
   ASSERT_NE(grid.get(0, 0, grid.getLocalSize()[2] + numGhostCells - 1), nullptr);
   ASSERT_EQ(grid.get(0, 0, grid.getLocalSize()[2] + numGhostCells), nullptr);
}
