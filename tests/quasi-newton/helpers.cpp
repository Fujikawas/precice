#ifndef PRECICE_NO_MPI

#include "helpers.hpp"
#include <iostream>

#include "precice/precice.hpp"
#include "testing/Testing.hpp"

/// tests for different QN settings if correct fixed point is reached
void runTestQN(std::string const &config, TestContext const &context)
{
  std::string meshName, writeDataName, readDataName;

  if (context.isNamed("SolverOne")) {
    meshName      = "MeshOne";
    writeDataName = "Data1";
    readDataName  = "Data2";
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    meshName      = "MeshTwo";
    writeDataName = "Data2";
    readDataName  = "Data1";
  }

  precice::Participant interface(context.name, config, context.rank, context.size);

  VertexID vertexIDs[4];

  // meshes for rank 0 and rank 1, we use matching meshes for both participants
  double positions0[8] = {1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.5};
  double positions1[8] = {2.0, 0.0, 2.0, 0.5, 2.0, 1.0, 2.0, 1.5};

  if (context.isNamed("SolverOne")) {
    if (context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    } else {
      interface.setMeshVertices(meshName, positions1, vertexIDs);
    }
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    if (context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    } else {
      interface.setMeshVertices(meshName, positions1, vertexIDs);
    }
  }

  interface.initialize();
  double inValues[4]  = {0.0, 0.0, 0.0, 0.0};
  double outValues[4] = {0.0, 0.0, 0.0, 0.0};

  int iterations = 0;

  while (interface.isCouplingOngoing()) {
    if (interface.requiresWritingCheckpoint()) {
    }

    double preciceDt = interface.getMaxTimeStepSize();
    interface.readData(meshName, readDataName, vertexIDs, preciceDt, inValues);

    /*
      Solves the following non-linear equations, which are extended to a fixed-point equation (simply +x)
      2 * x_1^2 - x_2 * x_3 - 8 = 0
      x_1^2 * x_2 + 2 * x_1 * x_2 * x_3 + x_2 * x_3^2 + x_2 = 0
      x_3^2 - 4 = 0
      x_4^2 - 4 = 0

      Analytical solutions are (+/-2, 0, +/-2, +/-2).
      Assumably due to the initial relaxation the iteration always converges to the solution in the negative quadrant.
    */

    if (context.isNamed("SolverOne")) {
      for (int i = 0; i < 4; i++) {
        outValues[i] = inValues[i]; // only pushes solution through
      }
    } else {
      outValues[0] = 2 * inValues[0] * inValues[0] - inValues[1] * inValues[2] - 8.0 + inValues[0];
      outValues[1] = inValues[0] * inValues[0] * inValues[1] + 2.0 * inValues[0] * inValues[1] * inValues[2] + inValues[1] * inValues[2] * inValues[2] + inValues[1];
      outValues[2] = inValues[2] * inValues[2] - 4.0 + inValues[2];
      outValues[3] = inValues[3] * inValues[3] - 4.0 + inValues[3];
    }

    interface.writeData(meshName, writeDataName, vertexIDs, outValues);
    interface.advance(1.0);

    if (interface.requiresReadingCheckpoint()) {
    }
    iterations++;
  }

  interface.finalize();

  // relative residual in config is 1e-7, so 2 orders of magnitude less strict
  BOOST_TEST(outValues[0] == -2.0, boost::test_tools::tolerance(1e-5));
  BOOST_TEST(outValues[1] == 0.0, boost::test_tools::tolerance(1e-5));
  BOOST_TEST(outValues[2] == -2.0, boost::test_tools::tolerance(1e-5));
  BOOST_TEST(outValues[3] == -2.0, boost::test_tools::tolerance(1e-5));

  // to exclude false or no convergence
  BOOST_TEST(iterations <= 20);
  BOOST_TEST(iterations >= 5);
}

/// tests for different QN settings if correct fixed point is reached mesh with empty partition
void runTestQNEmptyPartition(std::string const &config, TestContext const &context)
{
  std::string meshName, writeDataName, readDataName;

  if (context.isNamed("SolverOne")) {
    meshName      = "MeshOne";
    writeDataName = "Data1";
    readDataName  = "Data2";
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    meshName      = "MeshTwo";
    writeDataName = "Data2";
    readDataName  = "Data1";
  }

  precice::Participant interface(context.name, config, context.rank, context.size);

  VertexID vertexIDs[4];

  // meshes for rank 0 and rank 1, we use matching meshes for both participants
  double positions0[8] = {1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.5};

  if (context.isNamed("SolverOne")) {
    // All mesh is on primary rank
    if (context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    }
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    // All mesh is on secondary rank
    if (not context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    }
  }

  interface.initialize();
  double inValues[4]  = {0.0, 0.0, 0.0, 0.0};
  double outValues[4] = {0.0, 0.0, 0.0, 0.0};

  int iterations = 0;

  while (interface.isCouplingOngoing()) {
    if (interface.requiresWritingCheckpoint()) {
    }

    double preciceDt = interface.getMaxTimeStepSize();

    if ((context.isNamed("SolverOne") and context.isPrimary()) or
        (context.isNamed("SolverTwo") and (not context.isPrimary()))) {
      interface.readData(meshName, readDataName, vertexIDs, preciceDt, inValues);
    }

    /*
      Solves the following non-linear equations, which are extended to a fixed-point equation (simply +x)
      2 * x_1^2 - x_2 * x_3 - 8 = 0
      x_1^2 * x_2 + 2 * x_1 * x_2 * x_3 + x_2 * x_3^2 + x_2 = 0
      x_3^2 - 4 = 0
      x_4^2 - 4 = 0

      Analytical solutions are (+/-2, 0, +/-2, +/-2).
      Assumably due to the initial relaxation the iteration always converges to the solution in the negative quadrant.
    */

    if (context.isNamed("SolverOne")) {
      for (int i = 0; i < 4; i++) {
        outValues[i] = inValues[i]; // only pushes solution through
      }
    } else {
      outValues[0] = 2 * inValues[0] * inValues[0] - inValues[1] * inValues[2] - 8.0 + inValues[0];
      outValues[1] = inValues[0] * inValues[0] * inValues[1] + 2.0 * inValues[0] * inValues[1] * inValues[2] + inValues[1] * inValues[2] * inValues[2] + inValues[1];
      outValues[2] = inValues[2] * inValues[2] - 4.0 + inValues[2];
      outValues[3] = inValues[3] * inValues[3] - 4.0 + inValues[3];
    }

    if ((context.isNamed("SolverOne") and context.isPrimary()) or
        (context.isNamed("SolverTwo") and (not context.isPrimary()))) {
      interface.writeData(meshName, writeDataName, vertexIDs, outValues);
    }
    interface.advance(1.0);

    if (interface.requiresReadingCheckpoint()) {
    }
    iterations++;
  }

  interface.finalize();

  // relative residual in config is 1e-7, so 2 orders of magnitude less strict
  if ((context.isNamed("SolverOne") and context.isPrimary()) or
      (context.isNamed("SolverTwo") and (not context.isPrimary()))) {
    BOOST_TEST(outValues[0] == -2.0, boost::test_tools::tolerance(1e-5));
    BOOST_TEST(outValues[1] == 0.0, boost::test_tools::tolerance(1e-5));
    BOOST_TEST(outValues[2] == -2.0, boost::test_tools::tolerance(1e-5));
    BOOST_TEST(outValues[3] == -2.0, boost::test_tools::tolerance(1e-5));

    // to exclude false or no convergence
    BOOST_TEST(iterations <= 20);
    BOOST_TEST(iterations >= 5);
  }
}

void runTestQNBoundedValueSingleValue(std::string const &config, TestContext const &context)
{
  std::string meshName, writeDataName1, readDataName1;

  if (context.isNamed("SolverOne")) {
    meshName      = "MeshOne";
    writeDataName1 = "Data11";
    readDataName1  = "Data21";
    std::cout << "SolverOne" << std::endl;
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    meshName      = "MeshTwo";
    writeDataName1 = "Data21";
    readDataName1 = "Data11";
  }

  precice::Participant interface(context.name, config, context.rank, context.size);

  VertexID vertexIDs[1];

  // meshes for rank 0 and rank 1, we use matching meshes for both participants
  double positions0[2] = {1.0, 0.0};

  if (context.isNamed("SolverOne")) {
    if (context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    }
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    if (context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    }
  }

  interface.initialize();
  double inValues1[1]  = {0.1};
  double outValues1[1] = {0.00};

  int iterations = 0;

  while (interface.isCouplingOngoing()) {
    if (interface.requiresWritingCheckpoint()) {
    }

    double preciceDt = interface.getMaxTimeStepSize();
    interface.readData(meshName, readDataName1, vertexIDs, preciceDt, inValues1);

    if (context.isNamed("SolverOne")) {
      if (iterations == 0) {
        inValues1[0] = -0.2;
      }
      
        outValues1[0] = inValues1[0]; // only pushes solution through
      
    } else {
      int problem = 1;
      std::cout << "invalues1 in Solver2: " << inValues1[0] << std::endl;      
      switch (problem)
      {
      case 1:
        outValues1[0] = sin(inValues1[0]/ 0.50-0.5); // I
        break;
      case 2:
        break;
      case 3:
        break;
      case 4:
        break;
      default:
        break;
      }
    }

    interface.writeData(meshName, writeDataName1, vertexIDs, outValues1);

    interface.advance(1.0);

    if (interface.requiresReadingCheckpoint()) {
    }
    iterations++;
  }

  interface.finalize();
}


void runTestQNBoundedValueSimple(std::string const &config, TestContext const &context)
{
  std::string meshName, writeDataName1, readDataName1;

  if (context.isNamed("SolverOne")) {
    meshName      = "MeshOne";
    writeDataName1 = "Data11";
    readDataName1  = "Data21";
    std::cout << "SolverOne" << std::endl;
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    meshName      = "MeshTwo";
    writeDataName1 = "Data21";
    readDataName1 = "Data11";
  }

  precice::Participant interface(context.name, config, context.rank, context.size);

  VertexID vertexIDs[2];

  // meshes for rank 0 and rank 1, we use matching meshes for both participants
  double positions0[4] = {1.0, 0.0, 1.0, 1.2};

  if (context.isNamed("SolverOne")) {
    if (context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    }
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    if (context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    }
  }

  interface.initialize();
  double inValues1[2]  = {0.1, 0.2};
  double outValues1[2] = {0.00, 0.00};

  int iterations = 0;

  while (interface.isCouplingOngoing()) {
    if (interface.requiresWritingCheckpoint()) {
    }

    double preciceDt = interface.getMaxTimeStepSize();
    interface.readData(meshName, readDataName1, vertexIDs, preciceDt, inValues1);

    if (context.isNamed("SolverOne")) {
      if (iterations == 0) {
        inValues1[0] = -0.2;
        inValues1[1] = 0.2;
      }
      for (int i = 0; i < 2; i++) {
        outValues1[i] = inValues1[i]; // only pushes solution through
      }
    } else {
      int problem = 4;
      switch (problem)
      {
      case 1:
        outValues1[0] = sin(inValues1[0] * inValues1[1] / 0.10); // I
        outValues1[1] = cos(inValues1[0] * inValues1[1] / 0.250);
        break;
      case 2:
        outValues1[0] = sin(inValues1[0] * inValues1[1] / 0.5+0.12); // II
        outValues1[1] = sin(inValues1[0] * inValues1[1] * inValues1[1] / 0.25 + 0.4);
        break;
      case 3:
        outValues1[0] = sin(0.6 * inValues1[0] * inValues1[1] - 0.4 *inValues1[1] * inValues1[1] + 2); // III
        outValues1[1] = sin(inValues1[0] * inValues1[1] * inValues1[1] / 0.25 +0.4 * inValues1[0] * inValues1[1]  + 0.1);
        break;
      case 4:
        outValues1[0] = sin(6. * inValues1[0] * inValues1[1]+ 0.12); // IV
        outValues1[1] = sin(inValues1[0] * inValues1[1] * inValues1[1] / 0.25 +0.15);
        break;
      default:
        break;
      }
      std::cout << "outvalues1 in Solver2: " << outValues1[0] << "," << outValues1[1] << std::endl;
    }

    interface.writeData(meshName, writeDataName1, vertexIDs, outValues1);

    interface.advance(1.0);

    if (interface.requiresReadingCheckpoint()) {
    }
    iterations++;
  }

  interface.finalize();
}

void runTestQNBoundedValueComplex(std::string const &config, TestContext const &context)
{
  std::string meshName, writeDataName1, writeDataName2, writeDataName3, readDataName1, readDataName2, readDataName3;

  if (context.isNamed("SolverOne")) {
    meshName      = "MeshOne";
    writeDataName1 = "Data11";
    writeDataName2 = "Data12";
    writeDataName3 = "Data13";
    readDataName1  = "Data21";
    readDataName2  = "Data22";
    readDataName3  = "Data23";
    std::cout << "SolverOne" << std::endl;
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    meshName      = "MeshTwo";
    writeDataName1 = "Data21";
    writeDataName2 = "Data22";
    writeDataName3 = "Data23";
    readDataName1 = "Data11";
    readDataName2 = "Data12";
    readDataName3 = "Data13";
  }

  precice::Participant interface(context.name, config, context.rank, context.size);

  VertexID vertexIDs[6];

  // meshes for rank 0 and rank 1, we use matching meshes for both participants
  double positions0[12] = {1.0, 0.0, 1.0, 1.2, 1.0, 1.4, 1.0, 1.6, 1.0, 1.8, 1.0, 2.0};

  if (context.isNamed("SolverOne")) {
    if (context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    }
  } else {
    BOOST_REQUIRE(context.isNamed("SolverTwo"));
    if (context.isPrimary()) {
      interface.setMeshVertices(meshName, positions0, vertexIDs);
    }
  }

  interface.initialize();
  double inValues1[6]  = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  double inValues2[6]  = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  double inValues3[6]  = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  double outValues1[6] = {0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
  double outValues2[6] = {0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
  double outValues3[6] = {0.00, 0.00, 0.00, 0.00, 0.00, 0.00};

  int iterations = 0;

  while (interface.isCouplingOngoing()) {
    if (interface.requiresWritingCheckpoint()) {
    }

    double preciceDt = interface.getMaxTimeStepSize();
    interface.readData(meshName, readDataName1, vertexIDs, preciceDt, inValues1);
    interface.readData(meshName, readDataName2, vertexIDs, preciceDt, inValues2);
    interface.readData(meshName, readDataName3, vertexIDs, preciceDt, inValues3);

    if (context.isNamed("SolverOne")) {
      if (iterations == 0) {
        inValues1[0] = -0.3;
        inValues1[1] = 0.2;
        inValues1[2] = -0.9;
        inValues1[3] = 0.9;
        inValues1[4] = 0.5;
        inValues1[5] = -0.5;

        inValues2[0] = 0.3;
        inValues2[1] = 0.2;
        inValues2[2] = 0.9;
        inValues2[3] = 0.9;
        inValues2[4] = 0.5;
        inValues2[5] = 0.5;

        inValues3[0] = -0.3;
        inValues3[1] = 0.2;
        inValues3[2] = -0.9;
        inValues3[3] = 0.9;
        inValues3[4] = 0.5;
        inValues3[5] = -0.5;
      }
      for (int i = 0; i < 6; i++) {
        outValues1[i] = inValues1[i]; // only pushes solution through
        outValues2[i] = inValues2[i]; // only pushes solution through
        outValues3[i] = inValues3[i]; // only pushes solution through
      }
     } else {
      outValues1[0] = sin(inValues1[0] * inValues1[1] / 0.5+0.12); // II
      outValues1[1] = sin(inValues1[0] * inValues1[1] * inValues1[1] / 0.25 + 0.4);
      outValues1[2] = sin(0.6 * inValues1[2] * inValues1[3] - 0.4 *inValues1[3] * inValues1[3] + 2); // III
      outValues1[3] = sin(inValues1[2] * inValues1[3] * inValues1[3] / 0.25 +0.4 * inValues1[2] * inValues1[3]  + 0.1);
      outValues1[4] = sin(6. * inValues1[4] * inValues1[5]+ 0.12); // IV
      outValues1[5] = sin(inValues1[4] * inValues1[5] * inValues1[5] / 0.25 +0.15);

      outValues2[0] = sin(inValues2[0] * inValues2[1] / 0.5+0.12); // II
      outValues2[1] = sin(inValues2[0] * inValues2[1] * inValues2[1] / 0.25 + 0.4);
      outValues2[2] = sin(0.6 * inValues2[2] * inValues2[3] - 0.4 *inValues2[3] * inValues2[3] + 2); // III
      outValues2[3] = sin(inValues2[2] * inValues2[3] * inValues2[3] / 0.25 +0.4 * inValues2[2] * inValues2[3]  + 0.1);
      outValues2[4] = sin(0.6 * inValues2[4] * inValues2[5] - 0.4 *inValues2[5] * inValues2[5] + 2); // III
      outValues2[5] = sin(inValues2[4] * inValues2[5] * inValues2[5] / 0.25 +0.4 * inValues2[4] * inValues2[5]  + 0.1);
      // a fixed-problem without bounded data
      outValues3[0] = 1.0 / 3 * sin(inValues3[1]) + 1.0 / 4 * cos(inValues3[2]) + 1.0 / 5 * pow(inValues3[3], 2) + 1.0 / 6;
      outValues3[1] = 1.0 / 4 * cos(inValues3[0]) + 1.0 / 5 * sin(inValues3[4]) + 1.0 / 6 * sqrt(inValues3[5] + 2);
      outValues3[2] = 1.0 / 5 * sin(inValues3[1]) + 1.0 / 6 * cos(inValues3[3]) + 1.0 / 7 * log(inValues3[0] * inValues3[0] + 1);
      outValues3[3] = 1.0 / 6 * cos(inValues3[3]) + 1.0 / 7 * sin(inValues3[5]) + 1.0 / 8 * sqrt(std::abs(inValues3[4] - 1));
      outValues3[4] = 1.0 / 7 * sin(inValues3[4]) + 1.0 / 8 * cos(inValues3[1]) + 1.0 / 9 * exp(inValues3[2] - 1);
      outValues3[5] = 1.0 / 8 * cos(inValues3[5]) + 1.0 / 9 * sin(inValues3[0]) + 1.0 / 10 * log(inValues3[3] * inValues3[3] + 2);

      std::cout << "outvalues1 in Solver2: " << outValues1[0] << "," << outValues1[1]<<"," << outValues1[2]<<","<<outValues1[3]<<","<<outValues1[4]<<","<<outValues1[5] << std::endl;
      std::cout << "outvalues2 in Solver2: " << outValues2[0] << "," << outValues2[1]<<"," << outValues2[2]<<","<<outValues2[3]<<","<<outValues2[4]<<","<<outValues2[5] << std::endl;
      std::cout << "outvalues3 in Solver2: " << outValues3[0] << "," << outValues3[1]<<"," << outValues3[2]<<","<<outValues3[3]<<","<<outValues3[4]<<","<<outValues3[5] << std::endl;
    }

    interface.writeData(meshName, writeDataName1, vertexIDs, outValues1);
    interface.writeData(meshName, writeDataName2, vertexIDs, outValues2);
    interface.writeData(meshName, writeDataName3, vertexIDs, outValues3);

    interface.advance(1.0);

    if (interface.requiresReadingCheckpoint()) {
    }
    iterations++;
  }

  interface.finalize();
}
#endif
