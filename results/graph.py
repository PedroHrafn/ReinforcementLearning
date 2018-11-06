import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def getList(numbers):
    sum = 0
    averages = []
    dividers = []
    divider = 10000
    count = 0
    for i in numbers:
        i[0] = int(i[0])
        i[1] = float(i[1])
        if i[0] >= divider:
            averages.append(sum/count)
            sum = 0
            count = 0
            dividers.append(divider)
            divider += 10000
        else:
            sum += i[1]
            count += 1
    return averages


with open("MC_Results.txt") as f:
    numbers = [x.strip().split(':') for x in f.readlines()]
averages = getList(numbers)
plt.plot(averages)
with open("QL_Results.txt") as f:
    numbers = [x.strip().split(':') for x in f.readlines()]
averages = getList(numbers)
plt.plot(averages)
plt.show()
