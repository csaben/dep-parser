import re
import numpy
import pandas
import config

# tool for testing in browser
# https://regex101.com/r/fMy2g0/3


# collect a handful of util regex patterns to speed through parsing a txt file into a dataframe

# should end up looking kinda like this except way more general
# https://stackoverflow.com/questions/47982949/how-to-parse-complex-text-files-using-python/47984221#47984221:~:text=the%20column%20names

#regex to get first capital letter and up to next period
# r"[A-Z].{1}[^\.]*\."



if __name__ == "__main__":
    pattern = r"[A-Z].{1}[^\.]*\."
    re.compile(pattern)
    output = []
    with open(config.DATA, 'r') as f:
        ctr = 0
        for line in f:
            print(line)
            x = re.findall(pattern, line)
            ctr += 1
            output.extend(x)
            if output:
                print(output)
            if ctr > 100:
                break
