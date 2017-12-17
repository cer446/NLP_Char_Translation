#!/bin/bash

while getopts n:g: option
do
 case "${option}"
 in
 n) NAME=${OPTARG};;
 g) GPUID=${OPTARG};;
 esac
done

RUNDIR=/home/xc965/NLP/CHAR-NMT/src/$NAME

echo "GPUID:\n"
echo $GPUID

echo "RunDir: \n"
echo $RUNDIR

CUDA_VISIBLE_DEVICES=$GPUID python -u /home/xc965/NLP/CHAR-NMT/src/HashedNGramNMT.py


# while getopts n:g: option
# do
#  case "${option}"
#  in
#  n) NAME=${OPTARG};;
#  g) GPUID=${OPTARG};;
 
#  esac
# done

# RUNDIR=/home/xc965/NLP/CHAR-NMT/src/$NAME

# echo "GPUID: \n"
# echo $GPUID

# echo "RunDir: \n"
# echo $RUNDIR

# CUDA_VISIBLE_DEVICES=$GPUID python -u {RUNDIR}/${NAME} 


