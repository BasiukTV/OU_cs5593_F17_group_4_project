from datetime import datetime

# parses a simple subset of the ISO timeformat (no timezone, only datetime)
def parse_isotime(isostr):
    return datetime.strptime(isostr, "%Y-%m-%dT%H:%M:%S")
