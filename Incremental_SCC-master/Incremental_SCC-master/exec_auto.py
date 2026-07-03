import os
import subprocess

datasets_path='/hdd/thej_par_scc_datasets'
insert_methods=["5","6","11"]
# delete_methods=["8","10"]
num_threads_list=["72"]
abs_path =""

for dataset in os.listdir(datasets_path):
    if '.tar.gz' not in dataset:
        for filename in os.listdir(datasets_path+'/'+dataset):
            if 'refined_edges' in filename:
                abs_path = os.path.abspath(datasets_path+'/'+dataset+'/'+filename)
                subprocess.run(["./scc",abs_path,"1","2","-p"])
        for i in range(0,5):
            for filename in os.listdir(datasets_path+'/'+dataset):
                    if 'insert_edges' in filename:
                        abs_path = os.path.abspath(datasets_path+'/'+dataset+'/'+filename)
                        for method in insert_methods:
                            for num_threads in num_threads_list:
                                subprocess.run(["./scc",abs_path,num_threads,method])
                    # if 'delete_edges' in filename:
                    #     abs_path = os.path.abspath(datasets_path+'/'+dataset+'/'+filename)
                    #     for method in delete_methods:
                    #         for num_threads in num_threads_list:
                    #             subprocess.run(["./scc",abs_path,num_threads,method])
