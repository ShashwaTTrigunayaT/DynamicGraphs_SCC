def read_file(filename):
    dataset_info=[]
    with open(filename,'r') as file:
        data_info=[]
        for line in file:
            if 'data' in line:
                info=line.strip('\n').split('=')
                temp_info=info[1].split(' ')
                data_info=[]
                data_info.extend([temp_info[0],int(temp_info[1]),int(temp_info[2])])
            if 'running_time' in line:
                info=line.strip('\n').split('=')
                data_info.append(float(info[1]))
            if 'SCCs' in line:
                info=line.strip('\n').split('=')
                data_info.append(int(info[1]))
                dataset_info.append(data_info)
                            
    return dataset_info

def write_file(filename,dataset_info):
    lines=[]
    lines.append("Filename,method,threads,runtime,scc_count\n")
    for data in dataset_info:
        if data[1]!=2:
            str1=''
            for i in range(0,len(data)-1):
                str1=str1+str(data[i])+','
            str1=str1+str(data[len(data)-1])+'\n'
            lines.append(str1)

    with open(filename,'w') as f:
        f.writelines(lines)

dataset_info=read_file('output_may_27.txt')
write_file('req_info.txt',dataset_info)