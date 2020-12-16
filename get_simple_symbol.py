import os


if __name__=="__main__":
    simple_symbol=['+', '-', '\\time', '=']
    simple_path = 'simple_symbol.txt'
    tmp=[]
    with open("symbol.txt",'r') as file:
        lines = file.readlines()
        for line in lines:
            word=line.split('\t')[-1].rstrip('\n')
            if word in simple_symbol:
                tmp.append(line)

    with open(simple_path,'w') as file:
        for t in tmp:
            file.write(t)