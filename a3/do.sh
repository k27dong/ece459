#!/bin/bash

bold=$(tput bold)

normal=$(tput sgr0)

color_positive() {
    echo -e "\033[32m$1\033[0m"
}

color_negative() {
    echo -e "\033[31m$1\033[0m"
}

color_title() {
    echo -e "\033[33m$*\033[0m"
}

title() {
    echo -e ${bold}$(color_title ==========$*==========)${normal}
}

test_num=10
cpu_res=()
cuda_res=()
output_res=()

clear
title "Setup"

cd kernel
./setgcc
cd ../
cargo build --release

title "Start running ${test_num} tests"

for ((i = 1; i <= test_num; i++)); do
    echo -ne " Running test: ${i} / ${test_num}"\\r
    python generate.py
    cpu_time=$(./target/release/lab3 cpu input/cnn.csv input/in.csv output/out.csv |& grep -oE '[0-9]+ microseconds of actual work done' | cut -f 1 -d ' ')
    cuda_time=$(./target/release/lab3 cuda input/cnn.csv input/in.csv output/out_cuda.csv |& grep -oE '[0-9]+ microseconds of actual work done' | cut -f 1 -d ' ')
    cpu_res+=(${cpu_time})
    cuda_res+=(${cuda_time})
    comp_output=$(python compare.py)
    [[ ${comp_output} == "Comparison finished" ]] && output_res+=("✅") || output_res+=("❌")
done

echo
title "Comparing results"
echo " CPU  | CUDA  | DIFF  | COMPARE"
for ((i = 0; i < ${#cpu_res[@]}; i++)); do
    value1=${cpu_res[$i]}
    value2=${cuda_res[$i]}
    diff=$((value1 - value2))
    [[ diff < 0 ]] && diff_output=$(color_negative +$diff) || diff_output=$(color_positive $diff)
    echo "${value1} | ${value2} | ${diff_output} |   ${output_res[$i]}"
done

