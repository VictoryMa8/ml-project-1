import pandas as pd
import seaborn as sns

iris = sns.load_dataset('iris')

def main():
    # sort ascending by sepal length
    # print(iris.sort_values('sepal_length'))

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

    # print(iris)

    # make a new DataFrame that is a copy of an existing dataframe, use copy() function
    # almost always want to say deep=True
    # this tells copy() to make a deep copy it will copy both the structure and values of the og data
    iris2 = iris.copy(deep=True)

    # see the data with only specified columns (a subset)
    # print(iris2['species'])

    # multiple columns, surround columns with []
    print("Subset, species and petal_width: \n", iris2[['species', 'petal_width']])

    # iloc allows you to select subsets by row and column number
    print("Row 0: \n", iris2.iloc[0])

    # give iloc two numbers, then it will select the value of the corresponding row and column
    print("Row 0, column 0: \n", iris2.iloc[0, 0])

    # you can also use slices with iloc (rows 1-2 (0:2), columns 4-5 (3:5))
    print("Slice: \n", iris2.iloc[0:2, 3:5])

if __name__ == "__main__":
    main()