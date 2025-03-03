import math

def euclidean(point1, point2):
  distance = math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2)   # euclidean distance formula
  print(f"The distance between {point1['id']} and {point2['id']} is {round(distance, 2)}.") # print result

# homework 3
p1 = {'x': 2, 'y': 3, 'id': '1'}
p2 = {'x': 3, 'y': 2, 'id': '2'}
p3 = {'x': 2, 'y': -3, 'id': '3'}
p4 = {'x': 3, 'y': -2, 'id': '4'}
p5 = {'x': 1, 'y': -1.5, 'id': '5'}
p6 = {'x': 0, 'y': -1.5, 'id': '6'}

C1 = {'x': 2.5, 'y': -2.5, 'id': 'C1'}
C2 = {'x': 2.5, 'y': 2.5, 'id': 'C2'}
C3 = {'x': 0.5, 'y': -1.5, 'id': 'C3'}

# cluster 0
a0 = {'x': -1, 'y': 2, 'id': 'a0'}
b0 = {'x': -1, 'y': 1, 'id': 'b0'}
c0 = {'x': 1, 'y': 1, 'id': 'c0'}
d0 = {'x': 1, 'y': 2, 'id': 'd0'}

# cluster 1
a1 = {'x': -2, 'y': 2, 'id': 'a1'}
b1 = {'x': 2, 'y': 2, 'id': 'b1'}

def main():
  # question 4
  print("\nINTRACLUSTER:")
  euclidean(a1, b1)

  euclidean(a0, b0)
  euclidean(a0, c0)
  euclidean(a0, d0)
  euclidean(b0, c0)
  euclidean(b0, d0)
  euclidean(c0, d0)

  print("\nINTERCLUSTER:")
  euclidean(a1, a0)
  euclidean(a1, b0)
  euclidean(a1, c0)
  euclidean(a1, d0)

  euclidean(b1, a0)
  euclidean(b1, b0)
  euclidean(b1, c0)
  euclidean(b1, d0)

if __name__ == "__main__":
  main()