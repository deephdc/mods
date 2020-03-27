#!/bin/bash

for year in `seq 2018 2020`; do
	for month in `seq 1 12`; do
		month=$(printf '%02d' "$month")
		find . -name '*.tsv' | egrep "/$year/$month/" | sort | zip -9 "$year-$month.zip" -@
	done
done
