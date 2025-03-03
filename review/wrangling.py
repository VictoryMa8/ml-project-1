import pandas as pd
import seaborn as sns
import numpy as np

iris = sns.load_dataset('iris')
diamonds = sns.load_dataset('diamonds')

def main():
    # sort ascending by sepal length
    print(iris.sort_values('sepal_length'))

    '''
    use .set_index() to have a column like 'id' to be the index
    indexes technically aren't variables
    reset this index with .reset_index()

    inplace=True argument tells a function to change the data frame it acts on
    add ascending=False so it’s also sorted in descending order
    '''

    iris.sort_values(
        'petal_width',
        ascending=False,
        inplace=True
        )
    
    # make a new DataFrame that is a copy of an existing dataframe, use copy() function
    # almost always want to say deep=True
    # this tells copy() to make a deep copy it will copy both the structure and values of the og data
    iris2 = iris.copy(deep=True)
    iris2 = iris2.sort_index()

    # see the data with only specified columns (a subset)
    print(iris2['species'])

    # multiple columns, surround columns with []
    print("Subset, species and petal_width: \n", iris2[['species', 'petal_width']])

    # iloc allows you to select subsets by row and column number
    print("Row 0: \n", iris2.iloc[0])

    # give iloc two numbers, then it will select the value of the corresponding row and column
    print("Row 0, column 0: \n", iris2.iloc[0, 0])

    # you can also use slices with iloc (rows 1-2 (0:2), columns 4-5 (3:5))
    print("Slice: \n", iris2.iloc[0:2, 3:5])

    # use loc (not iloc) to subset
    # loc function allows you to subset using the data frame's indexes
    # confusingly works differently than iloc:
    print(iris2.iloc[0:3])
    print(iris2.loc[0:3])

    # loc makes more sense in my opinion
    print(iris2.loc[:, ['species', 'petal_width']])

    # subset by conditions (here, we print only virginica flowers)
    print(iris2[iris['petal_width'] >= 2.4])

    # store this condition gives us a 'series', you can store this and use it later
    best_petals = iris['petal_width'] >= 2.4
    print(iris2[best_petals])

    # loc works similarly, except we can specify to see certain columns
    print(iris2.loc[best_petals, ['species', 'petal_width']])

    # conditions are the same as R, == != > < >= <=, and ~ is "not"
    print(iris2[~best_petals])

    # there are rules for which operators are evaluated first
    # use parengtheses to make sure everything is evaluated in the order you want

    # use | or & and parentheses for a combination of conditions
    print(iris2[(iris2['petal_width'] >= 2.4) & (iris2['petal_length'] >= 6.0)])

    # query() can make this slightly easier
    # we put the conditional in one big string, we don't have to use quotes for variables
    print(iris2.query('(petal_width >= 2.4) & (petal_length >= 6.0)'))

    # if we want a categorical variable to be a certain value, we use ""
    print(iris2.query('(species == "virginica") & (petal_length >= 6.4)'))

    # usually loc is the best way to subset, otherwise query() is best if performance isn't an issue

    # chain functions like so
    big_virginicas = (
        iris2.query('(species == "virginica") & (petal_length >= 6.6)').
        copy(deep=True).
        sort_values('petal_length') # see the periods
    )
    print(big_virginicas)

    # change data like this
    big_virginicas['petal_length'] = big_virginicas['petal_length'] * 2
    print(big_virginicas)

    # create an indicator variable
    # is_huge indicates if the petal width is 2.2 or bigger
    big_virginicas['is_huge'] = (big_virginicas['petal_width'] >= 2.2)
    print(big_virginicas)

    # NaN is defined by the numpy package
    # detect missing values using isna() or notna()

    # where() takes three arguments: a condition, result if true, result if false
    big_virginicas['is_alright_lookin'] = np.where(
        big_virginicas['sepal_length'] >= 7.7,
        'alright_lookin',
        'not_alright'
    )

    print(big_virginicas)

    # group by 'cut' (quality), look at mean price of each group
    print(diamonds.groupby('cut')['price'].mean())

    # look at max, min, size, etc.
    print(diamonds.groupby('cut')['price'].max())
    print(diamonds.groupby('cut')['price'].min())
    print(diamonds.groupby('cut')['price'].size())

if __name__ == "__main__":
    main()