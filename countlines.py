import sys

file = sys.argv[1]

with open(file) as infile:
    print(len(infile.read().split("\n")))
