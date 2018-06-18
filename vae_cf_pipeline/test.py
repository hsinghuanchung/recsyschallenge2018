import sys


try:
    print(sys.argv[1])

except IndexError as e:
    print(e.args)