import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy

def euclidean(point1, point2):
  distance = math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2)   # euclidean distance formula
  print(f"The distance between {point1['id']} and {point2['id']} is {round(distance, 2)}.") # print result

shoppers = pd.read_csv("./online_shoppers_intention.csv", sep = ",")

def main():
  # subset1, less variables
  subset1_columns = ["Browser", "Region", "VisitorType", "Revenue"]
  subset1 = shoppers[subset1_columns]
  # group by visitor type
  print(subset1)

if __name__ == "__main__":
    main()