#/bin/bash
set -e

mkdir -p bin
make -j8
mv mercury ui_designer bin/

if [ "$#" -eq 0 ]; then
	echo 'Running main engine...'
	./bin/mercury
elif [ "$1" = "ud" ]; then
	echo 'Running UD (UI designer)...'
	./bin/ui_designer
fi
