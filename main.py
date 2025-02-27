import math
import pandas as pd

def euclidean(point1, point2):
  distance = math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2)   # euclidean distance formula
  print(f"The distance between {point1['id']} and {point2['id']} is {round(distance, 2)}.") # print result

pt1 ={'x': 20, 'y': 30, 'id': 'A1'}
pt2 = {'x': 40, 'y': 50, 'id': 'B1'}

def main():
    shoppers = pd.read_csv("./online_shoppers_intention.csv", sep = ",")
    
    # subset1, less variables
    subset1_columns = ["Administrative", "Month", "OperatingSystems", "Browser", "Region", "VisitorType", "Weekend", "Revenue"]
    subset1 = shoppers[subset1_columns]
    print(subset1)

    # subset2, even less
    subset2_columns = ["Browser", "Region", "VisitorType", "Revenue"]
    subset2 = subset1[subset2_columns]
    print(subset2)

    print(subset2.groupby(by = "VisitorType").count())

if __name__ == "__main__":
    main()