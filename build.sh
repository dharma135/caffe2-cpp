#module load cmake/3.10
module load caffe2-dev
#module load gcc/4.9.3
insDIR=`echo $PWD`
cd build
#echo $PWD
#insDIR=`$PWD/../`
CC=gcc CXX=g++ cmake3 .. -DCMAKE_INSTALL_PREFIX=$insDIR
make install
cd ..
