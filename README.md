import numpy as np

dataset = [
    [1,3,1.5],
    [0,2,1],
    [1,4,1.5],
    [0,3,1],
    [1,3.5,.5],
    [0,2,.5],
    [1,5.5,1],
    [0,1,1]
]

data = np.array([[dataset[i][1], dataset[i][2]] for i in range(len(dataset))])
target = np.array([dataset[i][0] for i in range(len(dataset))])

w = np.array([np.random.randn() for i in range(2)])
b = np.random.randn()
it = 10000
alpha = 0.01

def test(d,w,b):
    if sigmoid(np.dot([d[0],d[1]],w) + b) > .5:
        return 'Red'
    else:
        return 'Blue'

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def cost(i,o):
    return np.square(o-i)

def dcost(i,o):
    return 2*(o-i)
    
print(w,b)

for suichan in range(it):
    d = data

    z = np.dot(d,w) + b
    o = sigmoid(z)
    
    co = o - target
    c = co.sum()

    dc_do = co
    do_dz = dsigmoid(o)
    dc_dz = dc_do * do_dz
    d = data.T
    df = np.dot(d,dc_dz)

    w -= alpha * df
    for i in dc_dz:
        b -= alpha * i

print(w,b)
test([4.5,1],w,b) #Harusnya 'Red'
#Masalah yang muncul: 
#(1) kalo output blue muncul exp overflow [DONE]
#(2) Bias tidak keupdate
#(3) Masih sering salah nebak
