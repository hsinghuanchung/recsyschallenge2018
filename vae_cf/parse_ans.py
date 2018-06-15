f_ans = open('ans.out','r')
f_map = open('./data/mapping25','r')
f_out = open('task8.csv','w')

mapping = []

for line in f_map:
	line = line.strip().split()
	mapping.append(line[0])

for line in f_ans:
	line = [ int(_) for _ in line.strip().split()]
	print(len(line))	
	f_out.write("{0}".format(mapping[line[0]]))

	for data in line[1:]:
		f_out.write(",{0}".format(mapping[data]))
	f_out.write("\n")

