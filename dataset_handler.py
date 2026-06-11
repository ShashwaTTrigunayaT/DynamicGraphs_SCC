import os
import random

datasets_path='/hdd/thej_par_scc_datasets'
batch_sizes_list=[0.01,0.03,0.05,0.07,0.1,0.15]

def read_file(filename):
    edges_list=[]
    flag=0
    num_vertices=0
    max_vertex=0
    adj_list={}
    with open(filename,'r') as file:
        for line in file:
            if '%' not in line:
                edge=line.strip('\n').split(' ')
                if flag==1 and int(edge[0])!=int(edge[1]):
                    edges_list.append([int(edge[0]),int(edge[1])])
                    max_vertex=max(max_vertex,max(int(edge[0]),int(edge[1])))
                elif flag==0:
                    num_vertices=int(edge[0])
                flag=1
    
    for edge in edges_list:
        if edge[0] not in adj_list:
            adj_list[edge[0]]=[edge[1]]
        else:
            adj_list[edge[0]].append(edge[1])
    
    for key in adj_list:
        adj_list[key].sort()
    
    if num_vertices==max_vertex:
        print("Yes")
    
    return edges_list,adj_list,num_vertices

def write_file(filename,edges_list):
    lines=[]
    for edge in edges_list:
        lines.append(str(edge[0])+' '+str(edge[1])+'\n')
    with open(filename,'w') as f:
        f.writelines(lines)

def gen_dyn(edges_list,batch_size,adj_list,num_vertices):
    insert_edges=[]
    count1=0
    keys=list(adj_list.keys())
    random.shuffle(keys)
    for key in keys:
        if count1<=int(batch_size*len(edges_list)):
            ver_list=adj_list[key]
            for i in range(0,len(ver_list)-1):
                if count1<=int(batch_size*len(edges_list)):
                    diff=ver_list[i+1]-ver_list[i]
                    for j in range(1,diff):
                        if key!=ver_list[i]+j and count1<=int(batch_size*len(edges_list)):
                            if ver_list[i]+j in adj_list:
                                if key not in adj_list[ver_list[i]+j]:
                                    insert_edges.append([ver_list[i]+j,key])
                                    count1+=1
                            else:
                                insert_edges.append([ver_list[i]+j,key])
                                count1+=1
                        elif count1>int(batch_size*len(edges_list)):
                            break
                else:
                    break
        else:
            break

    delete_edges=random.sample(edges_list,int(batch_size*len(edges_list)))

    print(int(batch_size*len(edges_list)))
    print(len(insert_edges))

    return insert_edges,delete_edges

for dataset in os.listdir(datasets_path):
    if '.tar.gz' not in dataset:
        if not os.path.isfile(datasets_path+'/'+dataset+'/refined_edges.txt'):
            for filename in os.listdir(datasets_path+'/'+dataset):
                if filename==dataset+'.mtx':
                    print(filename)
                    edges_list,adj_list,num_vertices=read_file(datasets_path+'/'+dataset+'/'+filename)
                    write_file(datasets_path+'/'+dataset+'/refined_edges.txt',edges_list)
                    insert_edges_total,delete_edges_total=gen_dyn(edges_list,0.2,adj_list,num_vertices)
                    for batch_size in batch_sizes_list:
                        insert_edges=random.sample(insert_edges_total,int(batch_size*len(edges_list)))
                        delete_edges=random.sample(delete_edges_total,int(batch_size*len(edges_list)))
                        write_file(datasets_path+'/'+dataset+'/'+str(batch_size)+'_insert_edges.txt',insert_edges)
                        write_file(datasets_path+'/'+dataset+'/'+str(batch_size)+'_delete_edges.txt',delete_edges)
