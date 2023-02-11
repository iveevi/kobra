#!/bin/bash
lib=$1
echo 'Creating ArmaDA RTX plugin for library' $lib
dir=`dirname $lib`
name=`basename $lib`
name=${name%.*}
name=${name#lib}
echo 'Name:' $name

# Create the plugin
metadata=$dir/$name.json
full=$(readlink -f $lib)
echo 'Creating metadata file' $metadata
echo '{' > $metadata
echo '  "name": "'$name'",' >> $metadata
echo '  "library": "'$full'"' >> $metadata
echo '}' >> $metadata
