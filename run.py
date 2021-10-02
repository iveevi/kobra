import argparse

parser = argparse.ArgumentParser()

parser.add_argument("target", help="Database name")
parser.add_argument("-m", "--mode", help="Execution mode", default='')

args = parser.parse_args()

print(args)
