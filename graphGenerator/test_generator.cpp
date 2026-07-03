#include "graphGenerator.h"

int main()
{
    // Generate diameter graph
    

   
    // Generate LCC graph
    
   
    generate_lcc_graph_to_file(3000000,30,98000000);
    generate_lcc_graph_to_file(3000000,40,98000000);
    generate_lcc_graph_to_file(3000000,50,98000000);
    generate_lcc_graph_to_file(3000000,70,98000000);
    

    
    

    // Custom batches
    //generate_scc_aware_insert_batches("diameter_45 _1000000_100000.txt","Insertion_batch",{0.18f, 0.20f, 0.40f, 0.60f, 0.1f},0.70f);

    return 0;
}
