import statistics
f_out = open('ans.out')
f_ans = open('./data/pro_sg/test_te.csv')

arr=[]
for line_out in f_out:
	line_out = line_out.strip().split()
	line_out = set([ int(_) for _ in line_out])

	line_ans = f_ans.readline()
	line_ans = line_ans.strip().split()
	line_ans = set([ int(_) for _ in line_ans])

	arr.append(len(line_ans.intersection(line_out))/len(line_ans))

print(arr)
print(statistics.mean(arr))
