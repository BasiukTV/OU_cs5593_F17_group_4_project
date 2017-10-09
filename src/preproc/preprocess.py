import json, multiprocessing

DEFAULT_INPUT_DIRECTORY = "../../samples/data/raw/";
DEFAULT_THREADS_NUMBER = 5;

def pretty_print(obj):
    print(json.dumps(obj, sort_keys=True, indent=4, separators=(",", ": ")))

def process_day(month_and_day):
    (month, day) = month_and_day
    starcount = 0
    for hour in range (0,24):
        filename = "/scratch/timo/github/2016-{:02}-{:02}-{}.json".format(month, day, hour)
        try:
            with open(filename) as json_file:
                for line in json_file:
                    obj = json.loads(line)
                    t = obj["type"]
                    if t == "WatchEvent":
                        starcount += 1
        except IOError as er:
            print(er)
            pass

    print("{} new stars on 2017-{:02}-{:02}".format(starcount, month, day))


def process_month(month):
    out = pool.map(process_day, zip([month] * 31, range(1,32)))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Script for parsing input GitHub logs into application specific dataset.")
    parser.add_argument("-d", "--directory", help="Input logs file directory.", default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("-t", "--threads", help="Number of threads to use for processing.", default=DEFAULT_THREADS_NUMBER)
    args = parser.parse_args()

    print("Pre-processing started with:")
    print("Input Directory: {}".format(args.directory))
    print("Threads: {}".format(args.threads))

    pool = multiprocessing.Pool(int(args.threads))
    for month in range(1,4):
        process_month(month)
