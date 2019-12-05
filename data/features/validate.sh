#!/usr/bin/env bash

tsv=${1:-"./tsv"}


function next_day {
	date -j -v +1d -f "%Y/%m/%d" "$1" +"%Y/%m/%d"
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
		done
		d=$(next_day $d)
	done
	echo -e "--------------------------------------------------------------------------------\n"
done
