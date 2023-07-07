#!/bin/bash
clear

#### Helper Functions ####

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
    echo
    echo -e ${bold}$(color_title ==========$*==========)${normal}
}

to_ms() {
    local time_str="$1"
    local value=$(echo "$time_str" | sed 's/[a-z]*$//')
    local unit=$(echo "$time_str" | sed 's/[0-9.]*//')

    case "$unit" in
    "s") echo "$(echo "$value * 1000" | bc -l)ms" ;;
    "ms") echo "$value$unit" ;;
    *) echo "Invalid time unit: $unit" ;;
    esac
}

# https://stackoverflow.com/a/52209504
print_column() (
    file="${1:--}"
    if [ "$file" = - ]; then
        file="$(mktemp)"
        cat >"${file}"
    fi
    awk '
  FNR == 1 { if (NR == FNR) next }
  NR == FNR {
    for (i = 1; i <= NF; i++) {
      l = length($i)
      if (w[i] < l)
        w[i] = l
    }
    next
  }
  {
    for (i = 1; i <= NF; i++)
      printf "%*s ", w[i] + (i > 1 ? 1 : 0), $i
    print ""
  }
  ' "$file" "$file"
    if [ "$1" = - ]; then
        rm "$file"
    fi
)

#### VARIABLES ####
init_commit=9343d58f0546f6f0b234c57cb117c4d84b47d96e
current_branch_name=main
iter=1
old_result=()
student_result=()
match_result=()
input=()
print_result=""

declare -a num_idea_arr=(
    "80"
    "400"
    "800"
    "1000"
    # "8000"
    # "80000"
)

declare -a num_idea_gen_arr=(
    "2"
    "10"
    # "100"
)

declare -a num_pkgs_arr=(
    "4000"
    "8000"
    "20000"
    # "40000"
    # "400000"
    # "4000000"
)

declare -a num_pkg_gen_arr=(
    "6"
    "10"
    "12"
)

declare -a num_students_arr=(
    "6"
    "10"
    "20"
)

#### BEGIN SCRIPT ####
title "Building Input Combinations"
for num_idea in "${num_idea_arr[@]}"; do
    for num_idea_gen in "${num_idea_gen_arr[@]}"; do
        for num_pkgs in "${num_pkgs_arr[@]}"; do
            for num_pkg_gen in "${num_pkg_gen_arr[@]}"; do
                for num_students in "${num_students_arr[@]}"; do
                    curr="$num_idea $num_idea_gen $num_pkgs $num_pkg_gen $num_students"
                    input+=("$curr")
                done
            done
        done
    done
done

echo "Done"

title "Checking to the Initial Commit"
git -c advice.detachedHead=false checkout $init_commit

title "Build Original Executable"
cargo build --release

title "Running Tests (Original)"
for i in "${input[@]}"; do
    echo "Running test $iter: $i"
    result=$(hyperfine --warmup 3 -i "target/release/lab4 $i")
    mean=$(echo "$result" | awk '/Time \(mean ± σ\):/ {print $5 $6}')
    old_result+=($mean)
    iter=$((iter + 1))
done

iter=1

title "Checking to Main"
git checkout $current_branch_name

title "Building Current Executable"
cargo build --release

title "Running Tests (Optimized)"
for i in "${input[@]}"; do
    echo "Running test $iter: $i"
    result=$(hyperfine --warmup 3 -i "target/release/lab4 $i")
    mean=$(echo "$result" | awk '/Time \(mean ± σ\):/ {print $5 $6}')
    student_result+=($mean)
    iter=$((iter + 1))

    # Checksum Match Check
    output=$(target/release/lab4 $i)
    idea_gen=$(echo "$output" | grep -E "^Idea Generator:" | cut -d ' ' -f 3)
    student_idea=$(echo "$output" | grep -E "^Student Idea:" | cut -d ' ' -f 3)
    pkg_down=$(echo "$output" | grep -E "^Package Downloader:" | cut -d ' ' -f 3)
    student_pkg=$(echo "$output" | grep -E "^Student Package:" | cut -d ' ' -f 3)

    if [ "$idea_gen" = "$student_idea" ] && [ "$pkg_down" = "$student_pkg" ]; then
        match_result+=($(color_positive "Match"))
    else
        match_result+=($(color_negative "Wrong"))
    fi
done

title "Comparing results and time differences (green = good)"
for ((i = 0; i < ${#old_result[@]}; i++)); do
    value1=${old_result[i]}
    value2=${student_result[i]}
    match=${match_result[i]}
    value1_ms=$(to_ms "$value1")
    value2_ms=$(to_ms "$value2")
    arg=${input[i]}
    diff=$(echo "$value1_ms $value2_ms" | awk '{printf "%.1f", $2-$1}')
    speedup=$(echo "$value1_ms $value2_ms" | awk '{printf "%.1f", $1/$2}')
    if [ $(echo "$diff >= 0" | awk '{if ($0 >= 0) {print 1} else {print 0}}') -eq 1 ]; then
        output="$value1  $value2  $(color_negative "$speedup"x)  $match  $arg"
        print_result+=$"$output\n"
    else
        output="$value1  $value2  $(color_positive "$speedup"x)  $match  $arg"
        print_result+=$"$output\n"
    fi
done
echo -e "$print_result" | print_column

title "Done"
