import numpy as np

# 1. Weight > σ
# 2. Maximum weight
# 3. save the weights

a = np.loadtxt('results/re_gw.txt')
b = np.loadtxt('results/re_pw1.txt')
c = np.loadtxt('results/re_pw2.txt')
d = np.loadtxt('results/re_pw3.txt')


def maxcount(a, b, c, d):
    countg = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(len(a)):
        e = []
        e.append(a[i])
        e.append(b[i])
        e.append(c[i])
        e.append(d[i])
        m = max(e)
        if m == a[i]:
            countg += 1
        elif m == b[i]:
            count1 += 1
        elif m == c[i]:
            count2 += 1
        else:
            count3 += 1
    f = open("results/re_maximum_weight.txt", 'a')
    f.write("graph:" + str(countg) + ",pattern1:" + str(count1) + ",pattern2:" + str(count2) + ",pattern3:" + str(
        count3) + "\n")
    print("Maximum weight:", countg, count1, count2, count3)


def findValue(a, b, c, d, threshold):
    countg = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(len(a)):
        if a[i] >= threshold:
            countg = countg + 1
        if b[i] >= threshold:
            count1 = count1 + 1
        if c[i] >= threshold:
            count2 = count2 + 1
        if d[i] >= threshold:
            count3 = count3 + 1
    print(countg, count1, count2, count3)
    f = open("results/re_weights.txt", 'a')
    f.write(str(threshold) + ":" + str(countg) + " " + str(count1) + " " + str(count2) + " " + str(
        count3) + "\n")


maxcount(a, b, c, d)
findValue(a, b, c, d, 0.95)
findValue(a, b, c, d, 0.92)
findValue(a, b, c, d, 0.9)
findValue(a, b, c, d, 0.88)
findValue(a, b, c, d, 0.85)
findValue(a, b, c, d, 0.83)
findValue(a, b, c, d, 0.8)
