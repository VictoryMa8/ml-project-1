import pandas as pd
import seaborn as sns

diamonds = sns.load_dataset('diamonds')

def main():
    # group by 'cut' (quality), look at mean price of each group
    print(diamonds.groupby('cut')['price'].mean())

    # look at max, min, size, etc.
    print(diamonds.groupby('cut')['price'].max())
    print(diamonds.groupby('cut')['price'].min())
    print(diamonds.groupby('cut')['price'].size())

    print(diamonds)

if __name__ == "__main__":
    main()