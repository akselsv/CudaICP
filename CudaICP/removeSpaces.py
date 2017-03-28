import sys
if(len(sys.argv) != 2):
	print("Required input args: inputfile.txt")
	exit(0)
inputfile = str(sys.argv[1])


#file = open(outputfile,"w")

with open(inputfile,'r') as f:
	data = f.readlines()

with open(inputfile,'w') as file:
	for i in range(0,len(data)):
		file.write(data[i].lstrip())