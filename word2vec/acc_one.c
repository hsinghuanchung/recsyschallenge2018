/*

gcc acc_one.c -o acc_one

./acc_one [ground truth file]      [our ans file]
./acc_one truth_uri_800000_1000000 dis5_ans_800000_1000000

*/


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

const long long N = 510;//500

int main(int argc, char **argv) {
	FILE *f_truth;
	FILE *f_test;
	char file_name1[100], file_name2[100], pid[10], snum[10];
	char truthw[510][100], tmps[100], testw[510][100];
	int track_num, i, j, cnt;

	strcpy(file_name1, argv[1]);
	strcpy(file_name2, argv[2]);

  	f_truth = fopen(file_name1, "r");
  	f_test = fopen(file_name2, "r");

  	int playlist_num = 0, a, end_flag_truth = 0, end_flag_test = 0, zero_cnt = 0;
  	float total_acc = 0, max_acc = 0;
  	while (1) {
    	for (a = 0; a < N; a++) strcpy(truthw[a], " ");
    	for (a = 0; a < N; a++) strcpy(testw[a], " ");
    
    	fscanf(f_truth, "%s", pid);
  		fscanf(f_truth, "%s", snum);

  		//progress one playlist in f_truth
    	cnt = 0;
    	fscanf(f_truth, "%s", truthw[cnt]);
	    while(1){
	      if(strcmp(truthw[cnt], "ENDFILE") == 0){
	        end_flag_truth = 1;
	        break;
	      }
	      else if(strcmp(truthw[cnt], "NEXT") == 0)
	        break;
	      cnt++;
	      if(fscanf(f_truth, "%s", truthw[cnt]) != EOF)
	        continue;
	      else
	        break;
	    }
	    
	    //progress one playlist in f_test
    	cnt = 0;
    	fscanf(f_test, "%s", testw[cnt]);
	    while(1){
	      if(strcmp(testw[cnt], "ENDFILE") == 0){
	        end_flag_test = 1;
	        break;
	      }
	      else if(strcmp(testw[cnt], "NEXT") == 0)
	        break;
	      cnt++;
	      if(fscanf(f_test, "%s", testw[cnt]) != EOF)
	        continue;
	      else
	        break;
	    }

	    track_num = atoi(snum);
  		float sum = 0;
	  	for(i = 5; i < track_num; i++){
	  		for(j = 0; j < track_num; j++){
	  			if(strcmp(testw[j], truthw[i]) == 0){
	  				sum++;
	  				break;
	  			}
	  		}
	  	}

	  	//print acc result
	  	printf("sum= %f\ntrack_num=%d\n", sum, track_num);
  		float acc_one = sum / (track_num-5);
  		printf("acc for %d playlist= %f\n", playlist_num, acc_one);
  		printf("------------------------------\n");

  		total_acc += acc_one;
  		if(acc_one > max_acc)
  			max_acc = acc_one;
  		if(acc_one == 0.0)
  			zero_cnt++;
  		playlist_num++;


  		if(end_flag_test == 1)
  			break;
	} 	

	fclose(f_truth);
  	fclose(f_test);

  	printf("===Zero     Count: %d===\n", zero_cnt);
  	printf("===Max   Accuracy: %f===\n", max_acc);
  	printf("===Total Accuracy: %f===\n", (total_acc/playlist_num));


  	return 0;
}


