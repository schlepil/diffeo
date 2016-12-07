mkdir -p bin/Release
g++ -std=c++11 -O3 -fPIC -I/usr/include/eigen3 ../cpp/diffeoMethods.hpp ../cpp/diffeoMovements.hpp ../cpp/diffeoSearch.hpp ../cpp/FileVector.h ../cpp/thingsThatShouldExist.hpp ../cpp/main.cpp -o ./bin/Release/diffeoMethods
