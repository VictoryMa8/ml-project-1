import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import pingouin as pg



def euclidean(point1, point2):
  distance = math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2)   # euclidean distance formula
  print(f"The distance between {point1['id']} and {point2['id']} is {round(distance, 2)}.") # print result

shoppers = pd.read_csv("./online_shoppers_intention.csv", sep = ",")

def main():
  print(shoppers)

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

if __name__ == "__main__":
    main()