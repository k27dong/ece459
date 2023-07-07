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

# need to specify the initial commit hash
init_commit=bf68046b4c55c4901e9e7f0f3e65e9c0e23b9272
current_branch_name=main
iter=1
student_dir=student_ans
original_dir=original_ans
old_result=()
student_result=()
test_title=()
thread_num=100

declare -a maps_arg=(
    "--num-threads ${thread_num}"
    "--num-threads ${thread_num} --single-map"
)

# copied from readme, added escape characters
declare -a cmd_arr=(
    "--raw-hpc data/HPC.log --to-parse '58717 2185 boot_cmd new 1076865186 1 Targeting domains:node-D1 and nodes:node-[40-63] child of command 2176' --before-line '58728 2187 boot_cmd new 1076865197 1 Targeting domains:node-D2 and nodes:node-[72-95] child of command 2177' --after-line '58707 2184 boot_cmd new 1076865175 1 Targeting domains:node-D0 and nodes:node-[0-7] child of command 2175' --cutoff 106"
    "--raw-spark data/from_paper.log --to-parse '17/06/09 20:11:11 INFO storage.BlockManager: Found block rdd_42_20 locally' --before 'split: hdfs://hostname/2kSOSP.log:29168+7292' --after 'Found block' --cutoff 3"
    "--raw-linux data/Linux_2k.log --to-parse 'Jun 23 23:30:05 combo sshd(pam_unix)[26190]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.22.3.51 user=root' --before 'rhost=<> user=root' --after 'session opened' --cutoff 100"
    "--raw-hdfs data/HDFS_2k.log --to-parse '081109 204925 673 INFO dfs.DataNode\$DataXceiver: Receiving block blk_-5623176793330377570 src: /10.251.75.228:53725 dest: /10.251.75.228:50010' --before 'size <>' --after 'BLOCK* NameSystem.allocateBlock:'"
    "--raw-hpc data/HPC_2k.log --to-parse 'inconsistent nodesets node-31 0x1fffffffe <ok> node-0 0xfffffffe <ok> node-1 0xfffffffe <ok> node-2 0xfffffffe <ok> node-30 0xfffffffe <ok>' --before 'running running' --after 'configured out'"
    "--raw-hpc data/HPC.log --to-parse 'inconsistent nodesets node-31 0x1fffffffe <ok> node-0 0xfffffffe <ok> node-1 0xfffffffe <ok> node-2 0xfffffffe <ok> node-30 0xfffffffe <ok>' --before 'running running' --after 'configured out' --cutoff 106"
    "--raw-proxifier data/Proxifier_2k.log --to-parse '[10.30 16:54:08] chrome.exe - proxy.cse.cuhk.edu.hk:5070 close, 3637 bytes (3.55 KB) sent, 1432 bytes (1.39 KB) received, lifetime 00:01' --before 'proxy.cse.cukh.edu.hk:5070 HTTPS' --after 'open through' --cutoff 10"
    "--raw-healthapp data/HealthApp_2k.log --to-parse '20171223-22:15:41:672|Step_StandReportReceiver|30002312|REPORT : 7028 5017 150539 240' --before 'calculateAltitudeWithCache totalAltitude=240' --after 'onStandStepChanged 3601'"
    "--raw-healthapp data/HealthApp.log --to-parse '20171223-22:15:41:672|Step_StandReportReceiver|30002312|REPORT : 7028 5017 150539 240' --before 'calculateAltitudeWithCache totalAltitude=240' --after 'onStandStepChanged 3601' --cutoff 10"
)


title "Prepare directory for results"
rm -rf $original_dir
rm -rf $student_dir
mkdir $original_dir
mkdir $student_dir

title "Checking out to initial commit"
git checkout $init_commit

title "Building the current code"
cargo build --release

title "Getting sequential values and the time spent..."
for f in "${maps_arg[@]}"; do
    for e in "${cmd_arr[@]}"; do
        echo "Running test $iter"
        curr_arg="${e} ${f}"
        eval target/release/logram "$(echo $curr_arg)" > $original_dir/$iter.txt
        result=$(hyperfine -w 2 -r 2 -N --show-output "target/release/logram $curr_arg")
        mean=$(echo "$result" | awk '/Time \(mean ± σ\):/ {print $5 $6}')
        old_result+=($mean)
        iter=$((iter + 1))
    done
done
iter=1

title "Checking out to the current branch"
git checkout $current_branch_name

title "Building current code"
cargo build --release

title "Getting actual values and the current time spent..."
for f in "${maps_arg[@]}"; do
    for e in "${cmd_arr[@]}"; do
        echo "Running test $iter"
        curr_arg="${e} ${f}"
        eval target/release/logram "$(echo $curr_arg)" > $student_dir/$iter.txt
        result=$(hyperfine -w 2 -r 2 -N --show-output "target/release/logram $curr_arg")
        mean=$(echo "$result" | awk '/Time \(mean ± σ\):/ {print $5 $6}')
        student_result+=($mean)
        iter=$((iter + 1))
    done
done

title "Comparing results and time differences (green = good)"
for ((i = 0; i < ${#old_result[@]}; i++)); do
    if [[ $i -eq 9 ]]; then
        echo "================⬆️ Merged Map ⬆️================"
        echo "================⬇️ Single Map ⬇️================"
    fi
    value1=${old_result[i]}
    value2=${student_result[i]}
    diff=$(echo "$value1 $value2" | awk '{printf "%.1f", $2-$1}')
    output=$(python compare.py "$student_dir/$(($i + 1)).txt" "$original_dir/$(($i + 1)).txt")
    if [ $(echo "$diff >= 0" | awk '{if ($0 >= 0) {print 1} else {print 0}}') -eq 1 ]; then
        echo -e "$value1 | $value2 | $(color_negative +$diff) | $output"
    else
        echo -e "$value1 | $value2 | $(color_positive $diff) | $output"
    fi
done

title "Done"
