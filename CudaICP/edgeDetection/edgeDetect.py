import numpy as np
np.seterr(all='raise') #

cloud = input("Enter cloud name: ")
depth = np.empty((512,424))

x = 0
y = 0

with open(cloud) as f:
	for line in f:
		if x >= 512:
			x = 0
			y += 1
		floats = [float(x) for x in line.split()]
		depth[x,y] = floats[2];
		x += 1

v = np.vsplit(depth,128) #4 points wide
sdev = np.empty((512,424))

max = 0;
min = 100;

temp = 0


#Use standard deviation
for i in range (0,128):
	h = np.hsplit(v[i],106) #4 points broad
	for j in range (0,106):
		try:
			temp = np.std(h[j])
			if(np.std(h[j]) > max):
				max = np.std(h[j])
			if np.std(h[j]) < min:
				min = np.std(h[j])
			if abs(temp) > 10:
				temp = 0
			sdev[4*i:4*i+4,j*4:4*j+4] = temp
		except:
			temp = 0
			sdev[4*i:4*i+4,j*4:4*j+4] = temp


'''
x = 0
y = 0
#Use difference deviation
for i in range (0,128):
	h = np.hsplit(v[i],106) #4 points broad
	for j in range (0,106):
		try:
			temp = h[j][0,0]
			value = temp
			x= 0
			y=0
			for k in range (0,16):
				if x == 4:
					x = 0
					y += 1
				if abs(temp-h[j][x,y]) > 0.015:
					value = 0
				x += 1
			if(np.sum(h[j]) > max):
				max = np.sum(h[j])
			if np.sum(h[j]) < min and (np.isfinite(np.sum(h[j]))):
				min = np.sum(h[j])
			sdev[4*i:4*i+4,j*4:4*j+4] = value
		except:
			value = 0
			sdev[4*i:4*i+4,j*4:4*j+4] = value

print(max,min)
'''


x = 0
y = 0
file = open("edges_" + cloud,"w")
with open(cloud) as f:
	for line in f:
		if x >= 512:
			x = 0
			y += 1
		if (sdev[x,y] == 0.0 or sdev[x,y] >= 0.012) and ("-inf" not in line):
			file.write(line)
		x += 1
file.close()

file = open("remaining_" + cloud,"w")

x = 0
y = 0

with open(cloud) as f:
	for line in f:
		if x >= 512:
			x = 0
			y += 1
		if sdev[x,y] != 0.0 and sdev[x,y] < 0.012 and ("-inf" not in line):
			file.write(line)
		x += 1
file.close()
