dic = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Y','Z',
'W', '-', '.'}

file = open('dic.txt', 'w+',encoding="utf-8")
for i, j in enumerate(dic):
	file.write(str(i)+ "\t" +j)
	file.write("\n")

file.readline().rstrip("\n")
file.close()
