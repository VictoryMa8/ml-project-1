import pandas as pd
import os

def main():
    cwd = os.getcwd() # get current working directory
    print(cwd)

    # go back 1 folder, go to data folder, then get the data we want
    # tell function that the header is on row 1
    online_shoppers = pd.read_csv('../data/online_shoppers_intention.csv')
    print(online_shoppers)

    # skip the last five rows with skipfooter=5
    # skip first rows with skiprows=
    # say that na_values= are listed as NA, NAN, NULL, etc
    # use names= with a list to override variables names

if __name__ == "__main__":
    main()