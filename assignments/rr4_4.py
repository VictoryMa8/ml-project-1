import pandas as pd
import seaborn as sns

diamonds = pd.read_pickle('/output/diamonds2.pickle')

def main():
    diamonds['price'].groupby('quality').mean()

if __name__ == "__main__":
    main()