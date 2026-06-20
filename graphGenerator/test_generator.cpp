#include "graphGenerator.h"

int main()
{
    // Generate diameter graph
    generate_diameter_graph_to_file(45,1000000,10,100000);

    
    

    // Custom batches
    //generate_scc_aware_insert_batches("diameter_45 _1000000_100000.txt","Insertion_batch",{0.10f, 0.25f, 0.50f, 1.0f},0.70f);

    return 0;
}
