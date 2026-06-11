#include "gm.h"
#include <omp.h>
#include "scc.h"
#include "common_main.h"
#include "my_work_queue.h"

void delete_idea1(gm_graph &G,vector<vector<pair<int,int> > > rem_edges)
{
    int count1=0;
    // for(int i=0;i<rem_edges.size();i++)
    // {
        for(int j=0;j<rem_edges[5].size();j++)
        {
            int ver1=rem_edges[5][j].first;
            int ver2=rem_edges[5][j].second;
            if(ver1!=-1 && ver2!=-1)
            {
                G.add_edge(ver1,ver2);
                count1++;
            }
        }
    // }
    cout<<"num_new_edges:"<<G.num_edges()<<endl;
}
