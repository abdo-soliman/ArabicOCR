#!/bin/bash

usage()
{
    echo "random file sample generator
parameters:
        -s      --range-start   random range start value            DEFAULT 10
        -e      --range-end     random range end value              DEFAULT 10
        -n      --number        use if no range is required         DEFAULT 10
        -i      --input-dir     input directory to samlple from     DEFAULT ./scanned
        -o      --output-dir    output direcory to write sample in  DEFAULT ./sample
        -h      --help          print help"
}

inputDir="./scanned"
outputDir="./sample"
rangeStart=10
rangeEnd=10
number=10
noRange=1

while [ "$1" != "" ]; do
    case $1 in
        -s | --range-start )    shift
                                rangeStart=$1
                                noRange=0
                                ;;
        -e | --range-end )      shift
                                rangeEnd=$1
                                noRange=0
                                ;;
        -n | --number )         shift
                                number=$1
                                ;;
        -i | --input-dir )      shift
                                inputDir=$1
                                ;;
        -o | --output-dir )     shift
                                outputDir=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

if [ "$noRange" = "1" ]; then
    rangeStart=$number
    rangeEnd=$number
fi

if [ "$rangeEnd" -lt "$rangeStart" ]; then
    echo "invalid range"
    exit 1
fi

if [ -d $inputDir ]; then
    if [ -d $outputDir ]; then
        a=$(mktemp)
        find $inputDir -type f | shuf -n $(shuf -i$rangeStart-$rangeEnd -n1) >$a
        while IFS='' read -r l || [[ -n "$l" ]]; do
            cp "$l" $outputDir
        done <$a
    else
        mkdir $outputDir
        a=$(mktemp)
        find $inputDir -type f | shuf -n $(shuf -i$rangeStart-$rangeEnd -n1) >$a
        while IFS='' read -r l || [[ -n "$l" ]]; do
            cp "$l" $outputDir
        done <$a
    fi
else
    echo "input directory not found"
fi
