CXX=mpic++
CXXFLAGS= -O3 -std=c++11 -march=native -g -Wall

all: benchmark test ddtest operators

benchmark: benchmark.cpp ../fsgrid.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
operators: operators.cpp ../fsgrid.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
test: test.cpp ../fsgrid.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
ddtest: ddtest.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	-rm test benchmark ddtest operators
