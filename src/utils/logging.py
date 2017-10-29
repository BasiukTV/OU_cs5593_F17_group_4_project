from time import gmtime, strftime
from sys import stderr

def now():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def log(string, *args):
    print("{}: {}".format(now(), string).format(*args))

def elog(string, *args):
    print("{}: {}".format(now(), string).format(*args), file=stderr)
