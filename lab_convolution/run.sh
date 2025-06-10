#!/bin/bash

# Compile the cuDNN_filters.cpp file
g++ cuDNN_filters.cpp -o r -lcudnn -lcuda -lcudart

# Execute the compiled binary
./r

# Remove the binary after execution
rm r
