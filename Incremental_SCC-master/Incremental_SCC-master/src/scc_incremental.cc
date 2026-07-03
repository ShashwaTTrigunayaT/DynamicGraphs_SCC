#include "gm.h"
#include <omp.h>
#include "scc.h"
#include "common_main.h"
#include "my_work_queue.h"

void insert_idea1(gm_graph &G,vector<pair<int,int> > orig_edges, vector<pair<int,int> > insert_edges)
{
    for(int i=0;i<orig_edges.size();i++)
    {
        G.add_edge(orig_edges[i].first,orig_edges[i].second);
    }
    for(int i=0;i<insert_edges.size();i++)
    {
        G.add_edge(insert_edges[i].first,insert_edges[i].second);
    }
}

void insert_idea2(gm_graph &G,vector<pair<int,int> > scc_edges)
{
    for(int i=0;i<scc_edges.size();i++)
    {
        int ver1=scc_edges[i].first;
        int ver2=scc_edges[i].second;
        if(ver1+ver2 !=0)
            G.add_edge(ver1,ver2);
    }
}