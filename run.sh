. ./cmd.sh
. ./path.sh
set -e

tracknum=-1

. parse_options.sh || exit 1;
echo $tracknum
if [ $# != 0 -o "$tracknum" = "-1" ]; then
  echo "Usage: $0 --tracknum <1|2>"
  echo "  --tracknum <track number>         # number associated with the track to be run"
  exit 1;
fi

if [[ !( "$tracknum" == "1" || "$tracknum" == "2" || "$tracknum" == "2_den" ||
         "$tracknum" == "3" || "$tracknum" == "4" || "$tracknum" == "4_den") ]]; then
    echo "ERROR: Unrecognized track."
    exit 1
fi

./recipes/track${tracknum}/run.sh
