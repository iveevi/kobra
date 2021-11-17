#!/usr/bin/python3
import argparse
import os

# TODO: clean file and automate

# Execution modes
modes = {
    '': './',
    'gdb': 'gdb ',
    'mdb': './',
    'valgrind': 'valgrind '
}

parser = argparse.ArgumentParser()

# {TARGET} -m {EXECUTOR} - {THREADS}
parser.add_argument("target", help="Database name")
parser.add_argument("-m", "--mode", help="Execution mode", default='')
parser.add_argument("-j", "--threads",
                    help="Number of concurrent threads", type=int, default=8)

args = parser.parse_args()

if args.target == 'json':
    os.system('cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .')
    exit(0);

if args.mode in ['gdb', 'mdb', 'valgrind']:
    os.system('cmake -DCMAKE_BUILD_TYPE=Debug .')
elif args.mode == '':
    os.system('cmake .')
else:
    print('Invalid build mode...')
    exit(-1)

os.system('mkdir -p bin/')
if args.target == 'ud':
    ret = os.system(f'make -j{args.threads} ui_designer')
    if ret != 0:
        print('Failed compilation...')
        exit(-1)

    os.system('mv ui_designer bin/')

    executor = modes[args.mode]
    os.system(f'{executor}bin/ui_designer')
elif args.target == 'fb':
    ret = os.system(f'make -j{args.threads} file_browser')
    if ret != 0:
        print('Failed compilation...')
        exit(-1)

    os.system('mv file_browser bin/')

    executor = modes[args.mode]
    os.system(f'{executor}bin/file_browser')
elif args.target == 'cli':
    ret = os.system(f'make -j{args.threads} curses')
    if ret != 0:
        print('Failed compilation...')
        exit(-1)

    os.system('mv curses bin/')

    executor = modes[args.mode]
    os.system(f'{executor}bin/curses')
elif args.target == 'main':
    ret = os.system(f'make -j{args.threads} mercury')
    if ret != 0:
        print('Failed compilation...')
        exit(-1)

    os.system('mv mercury bin/')

    executor = modes[args.mode]
    os.system(f'{executor}bin/mercury')
