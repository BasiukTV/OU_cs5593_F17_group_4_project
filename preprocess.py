import multiprocessing

print("Job started")

def pretty_print(obj):
    print(json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': ')))

def process_day((month, day)):
    import json
    starcount = 0
    for hour in range (0,24):
        filename = '/scratch/timo/github/2016-{:02}-{:02}-{}.json'.format(month, day, hour)
        try:
            with open(filename) as json_file:
                for line in json_file:
                    obj = json.loads(line)
                    t = obj["type"]
                    if t == "WatchEvent":
                        starcount += 1
        except IOError:
            pass
    print("{} new stars on 2017-{:02}-{:02}".format(starcount, month, day))


def process_month(month):
    out = pool.map(process_day, zip([month] * 31, range(1,32)))

pool = multiprocessing.Pool(50)
for month in range(1,4):
    process_month(month)
