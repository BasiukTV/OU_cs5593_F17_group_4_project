from time import gmtime, strftime
from sys import stderr, stdout

def now():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def log(string, *args):
    print("{}: {}".format(now(), string).format(*args))
    stdout.flush()

def elog(string, *args):
    print("{}: {}".format(now(), string).format(*args), file=stderr)
    stderr.flush()

def progress(done, out_of, last=False):
    print("\r{}: Progress: {:.2f}%".format(now(), (100 * done) / out_of), end="" if not last else "\n")
