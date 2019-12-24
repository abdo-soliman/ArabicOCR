#!/bin/bash

RED=`tput setaf 1`
GREEN=`tput setaf 2`
YELLOW=`tput setaf 3`
RESET=`tput sgr0`

usage()
{
    echo "arabic ocr character dataset generator
parameters:
        -i      --img-dir       image directory to read image files from    DEFAULT ./DataSets/scanned
        -t      --text-dir      text directory to read text files from      DEFAULT ./DataSets/text
        -o      --output-dir    output direcory to write dataset in         DEFAULT ./dataSet
        -f      --file-type     output file type csv or img                 DEFAULT csv
        -l      --limit         number of images to be used -1 for all      DEFAULT -1
        -h      --help          print help"
}

imgDir="./DataSets/scanned"
textDir="./DataSets/text"
outputDir="./DataSets/dataSet"
limit=-1
fileType="csv"

# Declare an array of string with type
declare -a charArray=("ا" "ب" "ت" "ث" "ج" "ح" "خ" "د" "ذ" "ر" "ز" "ش" "س" "ص" "ض" "ط" "ظ" "ع" "غ" "ف" "ق" "ك" "ل" "م" "ن" "ه" "و" "ي" "ﻻ")

while [ "$1" != "" ]; do
    case $1 in
        -i | --img-dir )        shift
                                imgDir=$1
                                ;;
        -t | --text-dir )       shift
                                textDir=$1
                                ;;
        -o | --output-dir )     shift
                                outputDir=$1
                                ;;
        -f | --file-type )      shift
                                fileType=$1
                                ;;
        -l | --limit )          shift
                                limit=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

if [ -d "$imgDir" ]; then
    echo "${YELLOW}[INFO]: image directory found -> check on check text directory${RESET}"
    if [ -d "$textDir" ]; then
        echo "${YELLOW}[INFO]: text directory found -> check on output directory${RESET}"
        if [ -d "$outputDir" ]; then
            if [ "$fileType" == "img" ]; then
                echo "${YELLOW}[INFO]: output directory found -> check on letters' directories${RESET}"
                for val in ${charArray[@]}; do
                    if [ ! -d "$outputDir/$val" ]; then
                        echo "${YELLOW}[INFO]: $val directory not found -> create directory for $val${RESET}"
                        mkdir -p "$outputDir/$val"
                    fi
                done
            else
                echo "${YELLOW}[INFO]: output directory found${RESET}"
            fi
        else
            if [ "$fileType" == "img" ]; then
                echo "${YELLOW}[INFO]: output directory not found -> create output directory and letters' sub directories${RESET}"
                mkdir -p $outputDir
                for val in ${charArray[@]}; do
                    mkdir -p "$outputDir/$val"
                done
            else
                mkdir -p $outputDir
                echo "${YELLOW}[INFO]: output directory not found -> create output directory${RESET}"
            fi
        fi

        echo "${YELLOW}[INFO]: generating dataset...${RESET}"
        python segmentation.py -i $imgDir -t $textDir -d $outputDir -f $fileType -l $limit
        echo "${GREEN}[SUCCESS]: dataset generated${RESET}"
    else
        echo "${RED}[ERROR]: text directory not found${RESET}"
        exit 1
    fi
else
    echo "${RED}[ERROR]: image directory not found${RESET}"
    exit 1
fi
