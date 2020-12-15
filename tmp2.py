#### 不想改代码，直接写一个x1,y1,x2,y2转(x1,y1),(x2,y1),(x2,y2),(x1,y2)的
import os
txt_dir="train_1000/label"
txt_list=os.listdir(txt_dir)
for txt in txt_list:
    name=os.path.join(txt_dir,txt)
    with open(name,'r') as file:
        lines=file.readlines()
    new_lines=[]
    for line in lines:
        words=line.split(',')
        x1,y1,x2,y2=words[:4]
        x1, y1, x2, y2 = str(x1),str(y1),str(x2),str(y2)
        new_line=x1+','+y1+','+x2+','+y1+','+x2+','+y2+','+x1+','+y2+','+words[-1]
        new_lines.append(new_line)
    with open(name, 'w') as file:
        for new_line in new_lines:
            file.write(new_line)
