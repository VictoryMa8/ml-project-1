import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import pingouin as pg

def euclidean(point1, point2):
  distance = math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2)   # euclidean distance formula
  print(f"The distance between point1 and point2 is {round(distance, 2)}.") # print result

shoppers = pd.read_csv("./online_shoppers_intention.csv", sep = ",")

def main():
  print(shoppers)
  euclidean({'x': 2, 'y': 1}, {'x': 1, 'y': 2})
  euclidean({'x': 1, 'y': 3}, {'x': 1, 'y': 2})
  euclidean({'x': 2, 'y': 0}, {'x': 1, 'y': 2})
  euclidean({'x': 0, 'y': 2}, {'x': 1, 'y': 2})
  euclidean({'x': 3, 'y': 3}, {'x': 1, 'y': 2})

  '''
  sns.histplot(data = shoppers, x = 'VisitorType')
  # plt.show()

  sns.scatterplot(data = shoppers, x = 'Administrative', y = 'Informational')
  # plt.show()

  print(shoppers['Revenue'].value_counts())
  print(np.round(1908 / 10422, 4))

  # two sample t-test
  result1 = pg.ttest(shoppers['Administrative'], shoppers['Informational'], correction = True)
  # print(result1)

  # anova tests for visitor type
  print(pg.anova(dv= 'BounceRates', between = 'VisitorType', data = shoppers))
  print(pg.anova(dv = 'Administrative', between = 'VisitorType', data = shoppers))
  print(pg.anova(dv= 'ExitRates', between = 'VisitorType', data = shoppers))
  print(pg.anova(dv= 'PageValues', between = 'VisitorType', data = shoppers))

  # check for missing values in all columns
  missing_values = shoppers.isnull().sum()
  print('Missing values in each column:')
  print(missing_values)

  # check for 'strange' values
  strange_values = ['None', 'N/A', 'na', 'NA', 'n/a', 'null', 'NULL', '-', '?', 'unknown', 'UNKNOWN', 'missing']

  print('Check for strange placeholder values:')
  strange_counts = {}
  
  # count occurrences of each strange value
  for value in strange_values:
      total_count = 0
      for column in shoppers.columns:
          count = (shoppers[column].astype(str).str.strip() == value).sum()
          total_count += count
      
      if total_count > 0:
          strange_counts[value] = total_count

  print(f"Found strange values in dataset: {strange_counts}")

  # Print column data types
  print('Column data types:')
  print(shoppers.dtypes)
  '''

if __name__ == "__main__":
    main()