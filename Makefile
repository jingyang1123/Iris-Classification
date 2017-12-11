all:
	g++ -O3 -std=c++11 -o test_ffnn main.cpp

clean:
	rm *.csv
	rm test_ffnn
