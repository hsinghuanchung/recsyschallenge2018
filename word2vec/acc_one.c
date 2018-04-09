/*

calculat metrics:
1) R-precision 
2) Normalized discounted cumulative gain (NDCG)

gcc acc_one.c -o acc_one

./acc_one [ground truth file]      [our ans file]          [R_size][Playlist num] 
./acc_one truth_uri_800000_1000000 dis5_ans_800000_1000000 500 4000
./acc_one truth_uri_800000_800002 dis5_ans_800000_800002 500 2

*/


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>


const long long N = 510;

int main(int argc, char **argv) {
	FILE *f_truth;
	FILE *f_test;
	char file_name1[100], file_name2[100], pid[10], snum[10];
	char truthw[N][100], testw[N][100];
	int track_num, i, j, cnt;
	int R_size = atoi(argv[3]);
	int playlist_total = atoi(argv[4]);
	strcpy(file_name1, argv[1]);
	strcpy(file_name2, argv[2]);

  	f_truth = fopen(file_name1, "r");
  	f_test = fopen(file_name2, "r");

  	int playlist_num = 0, a = 0, zero_RP_cnt = 0, zero_NDCG_cnt = 0;
  	float total_RP = 0.0, max_RP = 0.0, acc_one = 0.0;
  	double total_NDCG = 0.0, max_NDCG = 0.0;
  	while (playlist_num < playlist_total) {
		//initialize
		for (a = 0; a < N; a++) strcpy(truthw[a], " ");
		for (a = 0; a < N; a++) strcpy(testw[a], " ");
    
		//progress one playlist in f_truth
		//fscanf(f_truth, "%s", pid);
		fscanf(f_truth, "%s", snum);
		track_num = atoi(snum);
		cnt = 0;
		while(cnt < track_num){
			fscanf(f_truth, "%s", truthw[cnt]);
			cnt++;
		}
	    
		//progress one playlist in f_test
		cnt = 0;
		while(cnt < R_size){
			fscanf(f_test, "%s", testw[cnt]);
			cnt++;
		}

		if(track_num <= 5){
			playlist_num++;
			continue;
		}

		///////////Metrics 1: calculate R-precision
		float sum = 0;
		for(i = 5; i < track_num; i++){
			for(j = 0; j < track_num; j++){
				if(strcmp(testw[j], truthw[i]) == 0){
					sum++;
					break;
				}
			}
		}
		acc_one = (sum / (track_num - 5));
		//cumulative total RP
		total_RP += acc_one;
		if(acc_one > max_RP)
			max_RP = acc_one;
		if(acc_one == 0.0)
			zero_RP_cnt++;
		//print RP result for one playlist
		//printf("Playlist: %d\n==R-precision: %f\n", playlist_num, acc_one);		
  		

		///////////Metrics 2: calculate NDCG
		double DCG = 0.0, IDCG = 1.0, NDCG = 0.0;
		int rel = 0;
		//for DCG
		for(j = 0; j < track_num; j++){
			if(strcmp(testw[i], truthw[j]) == 0){
				DCG = 1;//the same as "rel = 1"
				break;
			}
		}
		for(i = 1; i < R_size; i++){
			for(j = 0; j < track_num; j++){
				if(strcmp(testw[i], truthw[j]) == 0){
					rel = 1;
					break;
				}
			}
			if(rel){//i-th song of R is matched in ground truth 
				DCG += (1 / (log2((double)(i + 1))) );
			}
			rel = 0;
		}
		//for IDCG
		int match_GR = 0;//|G match R|
	  	for(i = 5; i < track_num; i++){
	  		for(j = 0; j < R_size; j++){
	  			if(strcmp(testw[j], truthw[i]) == 0){
	  				match_GR++;
	  				break;
	  			}
	  		}
	  	}
	  	for(i = 2; i <= match_GR; i++){
	  		IDCG += (1 / (log2((double)i)) );
	  	}
	  	//for NDCG
	  	if(match_GR == 0){
	  		NDCG = 0.0;
			zero_NDCG_cnt++;
		}
	  	else
	  		NDCG = (DCG / IDCG);
	  	//cumulative total NDCG
	  	total_NDCG += NDCG;
  		if(NDCG > max_NDCG)
  			max_NDCG = NDCG;
	  	//print NDCG result for one playlist
	  	//printf("==DCG: %f, IDCG: %f\n", DCG, IDCG);
	  	//printf("==NDCG: %f\n", NDCG);
	  	//printf("==match num: %d\n", match_GR);
	  	//printf("--------------------------------------\n"); 

  		playlist_num++;
	} 	

	fclose(f_truth);
  	fclose(f_test);

  	printf("playlist: %d\n", playlist_num);
  	printf("===Zero  R P   Count: %d===\n", zero_RP_cnt);
  	printf("===Max   R Precision: %f===\n", max_RP);
  	printf("===Aver  R Precision: %f===\n\n", (total_RP/playlist_num));
	
	printf("===Zero NDCG: %d===\n", zero_NDCG_cnt);
  	printf("===Max  NDCG: %f===\n", max_NDCG);
  	printf("===Aver NDCG: %f===\n", (total_NDCG/playlist_num));


  	return 0;
}


