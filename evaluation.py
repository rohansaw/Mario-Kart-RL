import re
import math
infile = r"./output.log"

important = []
keep_phrases = ["explained_variance",
              "step"]

def get_number(line):
    l= [int(s) for s in line.split() if s.isdigit()]
    numbers =  re.findall(r'\d+', line)
    if len(numbers) == 1:
        return int(numbers[0])
    if len(numbers) == 2:
        length = len(numbers[1]) - len(str(int(numbers[1])))
        print(length)
        return int(numbers[1]) / math.pow(10,len(str(numbers[1])))
    if len(numbers) == 3:
        print(numbers)
        return (int(numbers[0]) + (int(numbers[1]) / math.pow(10,len(str(numbers[1]))))) * math.pow(math.e, -1 * int(numbers[2]))

with open(infile) as f:
    f = f.readlines()

steps = []
losses = []
for line in f:
    for phrase in keep_phrases:
        if phrase in line:
            important.append(line)
            if phrase == "step":
                steps.append(get_number(line))
            if phrase == keep_phrases[0]:
                print(get_number(line))
                print(line)
                losses.append(get_number(line))
                if get_number(line) > 50:
                    steps.pop()
                    losses.pop()
            break
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes()
#x = np.linspace(0, 10, 1000)
plt.plot(steps, losses)
plt.show()


#print(important)