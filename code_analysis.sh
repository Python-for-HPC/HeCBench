#!/bin/bash

if [[ ! -f "rust-code-analysis-cli" ]]; then
    wget https://github.com/mozilla/rust-code-analysis/releases/download/v0.0.25/rust-code-analysis-linux-cli-x86_64.tar.gz >& /dev/null
    gunzip rust-code-analysis-linux-cli-x86_64.tar.gz >& /dev/null
    tar -xf rust-code-analysis-linux-cli-x86_64.tar >& /dev/null
    rm rust-code-analysis-linux-cli-x86_64.tar
fi

rm -rf code_analysis.txt
rm -rf code_metrics.txt

cd src

for f in adam floydwarshall haccmk hotspot3D lavaMD minibude
do
cd ${f}-omp
echo ${f}-omp >> ../../code_analysis.txt
find . -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -exec cat {} \; > ${f}-measure.all
mv ${f}-measure.all ${f}-measure.cpp
../../rust-code-analysis-cli -m -p ${f}-measure.cpp -O yaml >> ../../code_analysis.txt
rm ${f}-measure.cpp
cd ..
cd ${f}-pyomp
echo ${f}-pyomp >> ../../code_analysis.txt
find . -type f -name "*.py" -exec cat {} \; > ${f}-measure.all
mv ${f}-measure.all ${f}-measure.py
../../rust-code-analysis-cli -m -p ${f}-measure.py -O yaml >> ../../code_analysis.txt
rm ${f}-measure.py
cd ..
done

cd ..

while IFS= read -r; do
    # Print the next line
    printf '%s\n' "$REPLY" >> code_metrics.txt

    # Wait until a line starts with "metrics:"
    while ! [[ "$REPLY" =~ ^metrics: ]]; do
        read -r
    done

    # Print each desired metric until a blank line
    while [[ "$REPLY" != "" ]]; do
        trimmed_line="${REPLY#"${REPLY%%[![:space:]]*}"}"
        trimmed_line="${trimmed_line%"${trimmed_line##*[![:space:]]}"}"

        if [[ "$trimmed_line" =~ ^(volume:|difficulty:|effort:|time:) ]]; then
            printf '%s\n' "$REPLY" >> code_metrics.txt
        fi
        read -r
    done
done < "code_analysis.txt"

rm -rf code_analysis.txt
