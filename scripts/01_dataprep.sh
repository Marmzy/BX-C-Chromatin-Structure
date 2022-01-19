#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-v verbose] [-o output] [-t test] [-k kfold] [-l learn] [-i interpol] [-p perc]"
    echo " -h, --help       Print this help and exit"
    echo " -v, --verbose    Print verbose messages"
    echo " -o, --output     Name of output directory where data will be stored"
    echo " -t, --test       Size of the test dataset"
    echo " -k, --kfold      Number of folds to split the training dataset into"
    echo " -l, --learn      Type of algorithm (machine | deep)"
    echo " -i, --interpol   Interpolate missing value"
    echo " -p, --perc       Minimum percentage of barcodes to select cells"
    echo ""
    echo "Example: $0 -o data -t 0.2 -k 5 -l machine -i True -p 0.75"
    exit 1
}

#Parsing command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -v|--verbose) VERBOSE=true ;;
        -o|--output) OUTPUT="$2"; shift ;;
        -t|--test) TEST="$2"; shift ;;
        -k|--kfold) KFOLD="$2"; shift ;;
        -l|--learn) LEARN="$2"; shift ;;
        -i|--interpol) INTERPOL="$2"; shift ;;
        -p|--perc) PERC="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$VERBOSE" ]; then VALUE_V=false; else VALUE_V=true; fi;
if [ -z "$OUTPUT" ]; then OUT_DIR="data"; else OUT_DIR=$OUTPUT; fi;
if [ -z "$TEST" ]; then VALUE_T=0.2; else VALUE_T=$TEST; fi;
if [ -z "$KFOLD" ]; then usage "Number of kfolds is not specified"; else VALUE_K=$KFOLD; fi;
if [ -z "$LEARN" ]; then usage "Machine learning or Deep learning is not specified"; else VALUE_L=$LEARN; fi;
if [ -z "$INTERPOL" ]; then usage "Option to interpolate missing values"; else VALUE_I=$INTERPOL; fi;
if [ -z "$PERC" ]; then usage "Percentage of barcodes to select cells is not specified"; else VALUE_P=$PERC; fi;


#Making the output directory and subsequent subdirectories if it doesn't exist yet
if [ ! -e ${OUT_DIR} ]; then
    if [ "$VALUE_V" == "true" ]; then
        echo "Creating the data directory '${OUTPUT}'..."
    fi
    mkdir -p ${PWD%/*}/${OUT_DIR}
    mkdir -p ${PWD%/*}/${OUT_DIR}/raw
fi

#Downloading the raw data if it hasn't been done so before
if [ -z "$(ls -A ${PWD%/*}/${OUT_DIR}/raw)" ]; then
    if [ "$VALUE_V" == "true" ]; then
        echo -e "Downloading the raw data...\n"
    fi
    wget -P ${PWD%/*}/${OUT_DIR}/raw "https://zenodo.org/record/4741214/files/dnaData_exp1.csv"
    wget -P ${PWD%/*}/${OUT_DIR}/raw "https://zenodo.org/record/4741214/files/dnaData_exp2.csv"
    wget -P ${PWD%/*}/${OUT_DIR}/raw "https://zenodo.org/record/4741214/files/rnaData_exp1.csv"
    wget -P ${PWD%/*}/${OUT_DIR}/raw "https://zenodo.org/record/4741214/files/rnaData_exp2.csv"
fi

#Preparing the data
python3 ${PWD%/*}/src/data_prep.py \
        --verbose ${VALUE_V} \
        --output ${OUT_DIR} \
        --test ${VALUE_T} \
        --kfold ${VALUE_K} \
        --learn ${VALUE_L} \
        --interpol ${VALUE_I} \
        --perc ${VALUE_P}
