#ifndef PRECICE_NO_MPI

#pragma once

#include "testing/TestContext.hpp"

using namespace precice;
using precice::testing::TestContext;

void runTestQN(std::string const &config, TestContext const &context);

void runTestQNEmptyPartition(std::string const &config, TestContext const &context);

void runTestQNBoundedValueSingleValue(std::string const &config, TestContext const &context);

void runTestQNBoundedValueSimple(std::string const &config, TestContext const &context);

void runTestQNBoundedValueComplex(std::string const &config, TestContext const &context);

#endif
