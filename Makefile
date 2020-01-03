CXX=g++
INCLUDE=include/
SRC=src/
CXXFLAG+=-std=c++11 -ggdb
OPENCV =`pkg-config opencv --cflags --libs`

all: phe_opencv

phe_opencv: PHE_OpenCV.cpp libpheval.a
	${CXX} -I ${INCLUDE} ${CXXFLAG} $^ -o $@.out ${OPENCV}

libpheval.a: ${SRC}/evaluator5.o ${SRC}/hashtable5.o \
             ${SRC}/evaluator6.o ${SRC}/hashtable6.o \
             ${SRC}/evaluator7.o ${SRC}/hashtable7.o \
             ${SRC}/evaluator8.o ${SRC}/hashtable8.o \
             ${SRC}/evaluator9.o ${SRC}/hashtable9.o \
             ${SRC}/hash.o ${SRC}/hashtable.o ${SRC}/dptables.o ${SRC}/evaluator.o \
             ${SRC}/rank.o ${SRC}/7462.o
	ar rcs $@ $^

%.o: %.cc
	${CXX} -c ${CXXFLAG} -I${INCLUDE} $< -o $@

%.o: %.c
	${CC} -c ${CFLAGS} -I${INCLUDE} $< -o $@

clean:
	rm -rf *.o phe_opencv
