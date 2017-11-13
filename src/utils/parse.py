from datetime import datetime

# parses a simple subset of the ISO timeformat (no timezone, only datetime)
def parse_isotime(isostr):
    try:
        return datetime.strptime(isostr, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return datetime.strptime(isostr, "%Y/%m/%d %H:%M:%S")
