w = [0,0,0]
def f(x1,x2,p=True):
    teacher=int(not(x1 and x2))
    output=1 if w[0]+w[1]*x1+w[2]*x2>=0 else 0
    dt=teacher-output
    if p:
        print(x1,x2,output,teacher,w[0],w[1],w[2])
    w[0]+=dt
    w[1]+=dt*x1
    w[2]+=dt*x2

train = ((0,0),(0,1),(1,0),(1,1))

def h():
    for x1, x2 in train:
        teacher=int(not(x1 and x2))
        output=1 if w[0]+w[1]*x1+w[2]*x2>=0 else 0
        if teacher!=output:
            return False
    return True

# 随机尝试，直到找到 OK 的

from random import choice
def g():
    global w
    while True:
        w = [0,0,0]
        t = [(1,1),(0,0),(0,1)] + [choice(train) for _ in range(7)]
        for x1, x2 in t:
            f(x1, x2)
        if not h():
            print('fail')
        else:
            print('ok, final w = ', w)
            print('testcase = ', t)
            return tuple(w)
g()

# 绘图验证

import matplotlib.pyplot as plt
import numpy as np

def g2(): # 分界线所在也是红色的
    c, a, b = w
    
    x = np.linspace(-1, 2, 400)
    y = np.linspace(-1, 2, 400)
    X, Y = np.meshgrid(x, y)

    Z = a*X + b*Y + c

    plt.figure(figsize=(6,6))
    plt.contourf(X, Y, Z, levels=[0, Z.max()], colors='red', alpha=0.3)

    points = {'(0,0)': (0, 0), '(0,1)': (0, 1), '(1,0)': (1, 0), '(1,1)': (1, 1)}
    for label, (x, y) in points.items():
        color = 'blue' if label == '(1,1)' else 'red'
        plt.scatter(x, y, color=color)
        plt.text(x, y, label, fontsize=12, horizontalalignment='right')

    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.title('Graph of ax + by + c >= 0 with specific points marked')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

g2()

# 尝试能否找到其他解

# d = set()
# for i in range(100):
#     d.add(g())
# print(d)

# def g3():
#     global w
#     w = [0,0,0]
#     t = ((1, 1), (0, 0), (0, 1), (0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (1, 0), (0, 0))
#     for x1, x2 in t:
#         f(x1, x2)
#         g2()
# g3()

from random import randint
from collections import defaultdict
d = defaultdict(set)
def g4():
    global w
    while True:
        w = [0,0,0]
        t = [(1,1),(0,0),(0,1)] + [choice(train) for _ in range(randint(1,20))]
        for x1, x2 in t:
            f(x1, x2,False)
        if h():
            d[tuple(w)].add(tuple(t))
            return
for i in range(100):
    g4()
print(d.keys())
for k in d.keys():
    lens = set()
    for s in d[k]:
        lens.add(len(s))
    print(k, '->', sorted(list(lens)))
    
def g5(k):
    global w
    ss = set()
    for s in d[k]:
        w, t = [0, 0, 0], []
        for x1, x2 in s:
            oldw = tuple(w)
            f(x1, x2, False)
            if oldw != tuple(w):
                t.append((x1, x2))
        ss.add((len(t), tuple(t)))
    print(k, ss)
for k in d.keys():
    g5(k)
    
def g6(t):
    global w
    w = [0,0,0]
    for x1, x2 in t:
        f(x1, x2, False)
        print(w, sep= ' ')
    print()
g6(((1, 1), (0, 0), (0, 1), (1, 1), (0, 1), (1, 0), (1, 1), (1, 0), (1, 1), (1, 0)))
g6(((1, 1), (0, 0), (0, 1), (1, 1), (0, 1), (1, 0), (1, 1), (1, 0), (1, 1), (0, 1)))

def g7(t):
    global w
    w = [0,0,0]
    for x1, x2 in t:
        f(x1, x2, False)
    g2()

# g7(((1, 1), (0, 0), (0, 1), (1, 1), (1, 0)))
g7(((1, 1), (0, 0), (0, 1), (1, 1), (0, 1), (1, 0), (1, 1), (1, 0), (1, 1), (1, 0)))
g7(((1, 1), (0, 0), (0, 1), (1, 1), (0, 1), (1, 0), (1, 1), (1, 0), (1, 1), (0, 1)))
g7(((1, 1), (0, 0), (0, 1), (0, 1), (1, 0), (1, 1), (1, 0), (0, 0), (0, 0), (1, 0)))