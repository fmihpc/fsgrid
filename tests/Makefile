CXX=mpic++
# CXXFLAGS= -O3 -std=c++17 -ffast-math -march=native -g -Wall
CXXFLAGS= -O3 -std=c++17 -march=native -g -Wall
# CXXFLAGS= -O0 -std=c++17 -march=native -g -Wall

all: clean ddtest

benchmark: benchmark.cpp ../fsgrid.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
test: test.cpp ../fsgrid.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
ddtest: ddtest.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	-rm test ddtest benchmark
