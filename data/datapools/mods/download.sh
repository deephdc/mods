#!/bin/bash

while IFS= read -r  f
do
	of=$(echo "$f" | perl -p -E 's/.+([0-9]{4}-[0-9]{2}.zip).+/\1/g')
	wget "$f" -O "$of"
done < <(curl "https://seafile.sk/d/14094fcbdb3e422e9f8e/?p=/" | egrep -o -E '[0-9]{4}-[0-9]{2}.zip' | uniq | perl -p -E 's/(.+)/https:\/\/seafile.sk\/d\/14094fcbdb3e422e9f8e\/files\/?p=\/\1&dl=1/g')
