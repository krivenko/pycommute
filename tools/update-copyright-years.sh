#!/usr/bin/env sh

#
# Update copyright years in files
#

FIND=$(which find)
SED=$(which sed)

# Collect paths of files to be updated
set -o noglob
for PATTERN in "*.py" "*.cpp" "*.hpp"
do
    FILES="$FILES $($FIND . -name $PATTERN)"
done

# Write the new year string
YEARS_REGEX="([0-9]{4})-[0-9]{4}"
YEARS_UPDATED="\\1-$(date +%Y)"

COPYRIGHT_REGEX="Copyright \(C\) ${YEARS_REGEX}"
COPYRIGHT_UPDATED="Copyright \(C\) ${YEARS_UPDATED}"
for FILE in $FILES
do
    $SED -i -E "s/${COPYRIGHT_REGEX}/${COPYRIGHT_UPDATED}/g" $FILE
done

# docs/conf.py requires special treatment
sed -i -E "s/${YEARS_REGEX}, Igor Krivenko/${YEARS_UPDATED}, Igor Krivenko/g" \
       "docs/conf.py"
