#include "headers/transpose.h"
#include "headers/common.h"
#include "headers/opt_parser.h"

double mean(double X [], const int SIZE){
    double sum = 0;
    for (int i = 0; i < SIZE; i++){
        sum += X[i];
    }
    return (sum/SIZE);
}
double stdev(double X [], double mean, const int SIZE){
    double sum = 0;
    for (int i = 0; i < SIZE; i++){
        sum += pow(X[i] - mean, 2);
    }
    return sqrt(sum/SIZE);
}

int main(int argc, char* argv[]){
    
    // options to set benchmark, set in "run_benchmark.sh" file
    char method [8];
    int min_powerof2;           
    int max_powerof2;           
    int iterations_per_config;  
    int block_size;

    // check if input options are valid and assign to the passed variables
    process_benchmark_options(argc, argv, method, &min_powerof2, &max_powerof2, &iterations_per_config, &block_size);

    int num_configurations = max_powerof2 - min_powerof2 + 1;   //number of different data settings

    double Br_Bw;               // Bytes-read Bytes-wrote by the application    
    double exec_time [num_configurations][iterations_per_config];    // keep track of measured execution times 
    double effective_bandwidth [num_configurations][iterations_per_config];    // keep track of measured effective bandwidths

    char output_filename[100] = "";

    // init variables
    double mean_et, stdev_et;
    double mean_eb, stdev_eb;
    TIMER_DEF;

    // benchmarking of transpose_naive()
    if (strcmp(method, "naive") == 0 || strcmp(method, "all") == 0) {

        // Obtain Values
        printf("\nComputing statystics of : 'transpose_naive()' :");
        for (unsigned int i = 0; i < num_configurations; i++){

            const int ROWS = (int)pow(2,i+min_powerof2), COLS = ROWS;

            printf("\n   matrix size : 2^%d x 2^%d", i+min_powerof2, i+min_powerof2);

            for (unsigned int j = 0; j < iterations_per_config; j++){
                
                matrix mat = (matrix) malloc(ROWS * COLS * sizeof(matrix_element));
                fill_matrix(mat, ROWS, COLS);

                // Run the test
                TIMER_START;
                transpose_naive(mat, ROWS);
                TIMER_STOP;

                // Compute parameters
                exec_time[i][j] = TIMER_ELAPSED;

                Br_Bw = sizeof(matrix_element) * (ROWS*COLS - ROWS) * 2;
                effective_bandwidth[i][j] = ( Br_Bw / pow(10,9) ) / exec_time[i][j];
                
                free(mat);
            }
        }
        printf("\n");

        //Write results on a .csv file
        FILE *fp;
        sprintf(output_filename,"data/naive-version_%d-to-%d-steps_%s_%s.csv",min_powerof2,max_powerof2,VALUE(ADDITIONAL_OPTIM_FALG),VALUE(MATRIX_ELEM_DTYPE));
        fp = fopen(output_filename, "w");
        
        // Additional information about the data
        fprintf(fp,"Additional run info,,optimization flag,%s,Matrix element datatype,%s\n", VALUE(ADDITIONAL_OPTIM_FALG), VALUE(MATRIX_ELEM_DTYPE));

        // print column ID's
        fprintf(fp, "\nmatrix_size");
        for (unsigned int j = 0; j < iterations_per_config; j++){
            fprintf(fp, ",exec_time-%d,effective_bandwidth-%d", j, j);
        }
        fprintf(fp, ",mean_exec_time,stdev_exec_time,mean_effective_bandwidth,stdev_effective_bandwidth\n");

        // print data
        for (unsigned int i = 0; i < num_configurations; i++){
            fprintf(fp, "2^%d", i+min_powerof2);

            for (unsigned int j = 0; j < iterations_per_config; j++){
                fprintf(fp, ",%f,%f", exec_time[i][j], effective_bandwidth[i][j]);
            }
            
            mean_et = mean(exec_time[i], iterations_per_config);
            mean_eb = mean(effective_bandwidth[i], iterations_per_config);
            stdev_et = stdev(exec_time[i], mean_et, iterations_per_config);
            stdev_eb = stdev(effective_bandwidth[i], mean_eb, iterations_per_config);
            
            fprintf(fp, ",%f,%f,%f,%f\n",mean_et, stdev_et, mean_eb, stdev_eb);
        }
        fclose(fp);
    }


    // benchmarking of transpose_blocks()
    if (strcmp(method, "blocks") == 0 || strcmp(method, "all") == 0) {

        // block size is transformed in the corresponding power of 2
        const int BLK_SIZE = (int) pow(2,block_size);

        // Obtain Values
        printf("\nComputing statystics of : 'transpose_blocks()' :");
        for (unsigned int i = 0; i < num_configurations; i++){
            
            const int ROWS = (int)pow(2,i+min_powerof2), COLS = ROWS;
            printf("\n   matrix size : 2^%d x 2^%d", i+min_powerof2, i+min_powerof2);

            for (unsigned int j = 0; j < iterations_per_config; j++){
                matrix mat = (matrix) malloc(ROWS * COLS * sizeof(matrix_element));
                fill_matrix(mat, ROWS, COLS);

                // Run the test
                TIMER_START;
                transpose_blocks(mat, ROWS, BLK_SIZE);
                TIMER_STOP;

                // Compute parameters
                exec_time[i][j] = TIMER_ELAPSED;

                Br_Bw = sizeof(matrix_element) * (ROWS*COLS - ROWS) * 2;
                effective_bandwidth[i][j] = ( Br_Bw / pow(10,9) ) / exec_time[i][j];
                
                free(mat);
            }
        }
        printf("\n\n");

        //Write results on a .csv file
        FILE *fp;
        sprintf(output_filename,"data/blocks-version_%d-to-%d-steps_%d-blocksize_%s_%s.csv",min_powerof2,max_powerof2,block_size,VALUE(ADDITIONAL_OPTIM_FALG),VALUE(MATRIX_ELEM_DTYPE));
        fp = fopen(output_filename, "w");
        
        // Additional information about the data
        fprintf(fp,"Additional run info,,block size,2^%d x 2^%d,optimization flag,%s,Matrix element datatype,%s\n", block_size, block_size, VALUE(ADDITIONAL_OPTIM_FALG), VALUE(MATRIX_ELEM_DTYPE));

        // print column ID's
        fprintf(fp, "\nmatrix_size");
        for (int j = 0; j < iterations_per_config; j++){
            fprintf(fp, ",exec_time-%d,effective_bandwidth-%d", j, j);
        }
        fprintf(fp, ",mean_exec_time,stdev_exec_time,mean_effective_bandwidth,stdev_effective_bandwidth\n");

        // print data
        for (int i = 0; i < num_configurations; i++){
            fprintf(fp, "2^%d", i+min_powerof2);

            for (int j = 0; j < iterations_per_config; j++){
                fprintf(fp, ",%f,%f", exec_time[i][j], effective_bandwidth[i][j]);
            }
            
            mean_et = mean(exec_time[i], iterations_per_config);
            mean_eb = mean(effective_bandwidth[i], iterations_per_config);
            stdev_et = stdev(exec_time[i], mean_et, iterations_per_config);
            stdev_eb = stdev(effective_bandwidth[i], mean_eb, iterations_per_config);
            
            fprintf(fp, ",%f,%f,%f,%f\n",mean_et, stdev_et, mean_eb, stdev_eb);
        }
        fclose(fp);
    }
    
    return(0);
}
