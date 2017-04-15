Added new parameter file (../parameters/diffeoPars.xml) which is parsed using tinyxml (sudo apt-get install libtinyxml-dev)

To build the interface you need cython (sudo -H pip3 install cython) and then run 
python3 setup.py build_ext --inplace

Then the example "exampleCarlosPyEigen.py" should work.
