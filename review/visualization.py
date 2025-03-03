import pandas as pd
import seaborn as sns
import plotnine as p9

diamonds = sns.load_dataset('diamonds')

# drop variables, need to specify axis=1 when we're dropping columns
diamonds = diamonds.drop(
    [
        'table',
        'x',
        'y',
        'z'
    ],
    axis=1
)

# we can drop observations (rows) by specifying index and axis=0 for rows
diamonds = diamonds.drop(53930, axis=0)

# change column names
diamonds = diamonds.rename(
    columns = {
        'cut': 'quality'
    }
)

def main():
    # locate a certain observation by index
    # print(diamonds.loc[5])

    # look at variable levels
    # print(diamonds['quality'].value_counts())

    # create indicator variable
    diamonds['best'] = (diamonds['quality'] == "Ideal")

    # convert strings to categories
    diamonds[['quality', 'color']] = (
        diamonds[['quality', 'color']].
        astype('category')
    )

    # look at data types
    # print(diamonds.dtypes)

    # understand the distribution of a continous variable
    print(diamonds['price'].describe())

    '''
    use plotnine (like ggplot2 from R for visualizations)
    (p9.ggplot(diamonds, p9.aes(x = 'price')) + p9.geom_histogram(binwidth = 1000)).save('plot1.png')

    (p9.ggplot(diamonds, p9.aes(x = 'quality')) + p9.geom_bar()).save('plot2.png')
    '''

    # save a cleaned and changed data frame as a pickle
    diamonds.to_pickle('diamonds2.pickle')
    
if __name__ == "__main__":
    main()