#!/usr/bin/env bash

tsv=${1:-"./tsv"}


function next_day {
	date -j -v +1d -f "%Y/%m/%d" "$1" +"%Y/%m/%d"
}

regex="^.+-s([0-9]+)([smh])\.tsv"
function datapool_lines {
	if [[ "$1" =~ $regex ]]; then
		slide="${BASH_REMATCH[1]}"
		unit="${BASH_REMATCH[2]}"
		#echo "s:$slide, u:$unit"
		if [[ "$unit" == "s" ]]; then
			echo $(((24*60*60)/$slide))
		elif [[ "$unit" == "m" ]]; then
			echo $(((24*60)/$slide))
		elif [[ "$unit" == "h" ]]; then
			echo $((24/$slide))
		else
			echo $((-1))
		fi
	fi
}

echo -e "FEATURES DIR:\n$tsv\n"

# resolve protocols
protocols=$(find "$tsv" -depth 1 -type d | xargs basename | sort)
echo -e "PROTOCOLS:\n$protocols\n"

# resolve datapools
datapools=$(find ./tsv -type f -name '*.tsv' | xargs basename | sort | uniq)
echo -e "DATAPOOLS:\n$datapools\n"

# for each protocol
for protocol in $protocols; do
	beg=$(find "$tsv/$protocol" -type d -depth 3 | sort | head -n 1 | perl -p -E 's/.*(\d{4}\/\d{2}\/\d{2})$/\1/g')
	end=$(find "$tsv/$protocol" -type d -depth 3 | sort -r | head -n 1 | perl -p -E 's/.*(\d{4}\/\d{2}\/\d{2})$/\1/g')
	echo -e "PROTOCOL: $protocol"
	echo -e "BEG: $beg"
	echo -e "END: $end"
	echo -e "--------------------------------------------------------------------------------"
	d=$beg
	header=""
	protocol_missing=$((0))
	protocol_unexpected=$((0))
	protocol_lines=$((0))
	protocol_lines_expected=$((0))
	while [[ "$d" < "$end" ]]; do
		for datapool in $datapools; do
			f="$tsv/$protocol/$d/$datapool"
			if [ ! -f "$f" ]; then
				echo "ERROR: missing file '$f'"
				continue
			fi
			h=$(cat "$f" | head -n 1)
			if [[ "$header" == "" ]]; then
				header="$h"
				echo -e "--------------------------------------------------------------------------------"
				echo -e "HEADER: '$header'"
				echo -e "--------------------------------------------------------------------------------"
				continue
			fi
			if [[ "$h" != "$header" ]]; then
				echo "ERROR: header change in '$f'"
			fi
			header="$h"
			flines=$(tail -n +2 "$f" | wc -l)
			protocol_lines=$(($protocol_lines+$flines))
			dplines=$(datapool_lines "$datapool")
			protocol_lines_expected=$(($protocol_lines_expected+$dplines))
			if (( $dplines == -1 )); then
				echo "ERROR: unsupported window+slide: $datapool"
			elif (( $flines < $dplines )); then
				missing=$(($dplines-$flines))
				protocol_missing=$(($protocol_missing+$missing))
				echo -e "ERROR: $f\tmissing:$missing\t$flines/$dplines"
			elif (( $flines > $dplines )); then
				unexpected=$(($flines-$dplines))
				protocol_unexpected=$(($protocol_unexpected+$unexpected))
				echo -e "ERROR: $f\tunexpected:$unexpected\t$flines/$dplines"
			fi
		done
		d=$(next_day $d)
	done
	echo "PROTOCOL MISSING LINES:    $protocol_missing"
	echo "PROTOCOL UNEXPECTED LINES: $protocol_unexpected"
	echo "PROTOCOL LINES:            $protocol_lines"
	echo "PROTOCOL LINES EXPECTED:   $protocol_lines_expected"
	echo -e "--------------------------------------------------------------------------------\n"
done
