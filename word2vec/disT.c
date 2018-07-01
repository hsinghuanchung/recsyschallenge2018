/*

gcc disT.c -o disT -lm

./disT vectors.bin   [input file name]                 [output file name]              [playlist num]   
./disT ../vectors12.bin title_5input_800000_804000 dis5_title_800000_804000 4000
./disT ../vectors13.bin title_5input_800000_804000 dis5_title_800000_804000_13 4000

./disT ../vectors12.bin title_5input_800000_804000 dis5_title_800000_804000_a 4000


./disT ../vectorsB.bin task3_input 1000_1999.csv 4000
./disT ../vectorsB.bin task5_input 3000_3999.csv 4000
./disT ../vectorsB.bin task2_input 9000_9999.csv 4000

./disT ../vectorsB.bin task9_input_subtop3 7000_7999.csv 1000

*/


#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <malloc.h>
#include <stdlib.h>

const long long max_size = 2000;         // max length of strings
const long long N = 500;//500             // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries
const long long i_num = 5;


//////This is used in my own way of calculating the weight of five distance//////
struct average{
  char aword[50];
  float adist, cnt, final_dist;
};
typedef struct average Average;

int cmpfunc(const void *data1, const void *data2){
  Average *ptr1 = (Average *)data1;
  Average *ptr2 = (Average *)data2;
  if(ptr1->final_dist > ptr2->final_dist)
    return -1;
  else if(ptr1->final_dist < ptr2->final_dist)
    return 1;
  else
    return 0;
}


int main(int argc, char **argv) {
  FILE *f;
  FILE *f_test;
  FILE *f_for_acc;
  FILE *f_top500;
  char st1[max_size];
  char *bestw[10][N];
  char file_name[max_size], st[510][max_size], file_name2[max_size], file_name3[max_size];
  float dist, len, bestd[10][N], vec[max_size];
  long long words, size, a, b, c, d, cn, bi[300];
  char ch;
  float *M;
  char *vocab;
  int playlist_total = atoi(argv[4]);


  long long tmp = N * i_num;
  Average aver[tmp];


  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));

  int i;
  for (i = 0; i < 10; i++){
    for (a = 0; a < N; a++) 
      bestw[i][a] = (char *)malloc(max_size * sizeof(char));
  }

  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);


  ////start testing
  strcpy(file_name2, argv[2]);
  f_test = fopen(file_name2, "r");
  if (f_test == NULL) {
    printf("Input file2 not found\n");
    return -1;
  }

  strcpy(file_name3, argv[3]);//output file name

  int input_num = 0, check = 0, gc = 0, five_not_match = 0;
  char tmp_s[max_w];

  f_for_acc = fopen(file_name3, "w");


  

  char spid[7];
  printf("into check loop\n");
  while (check < playlist_total) {
    for (a = 0; a < N; a++) bestd[input_num][a] = 0;
    for (a = 0; a < N; a++) bestw[input_num][a][0] = 0;
    //printf("Enter %d word: ", input_num+1);
    //printf("Enter word or sentence (EXIT to break): ");

    fscanf(f_test, "%s", spid);

    a = 0;
    while(1){
      fscanf(f_test, "%s", st[a]);
      if(strcmp(st[a], "NEXT") == 0){
        break;
      }
      //printf("gc:%d, a:%lld\n", gc, a);
      a++;
      gc++;
    }
    
    cn = a;

    if(check % 100 == 0)
      printf("check: %d\n", check);
    check++;

    //count the position of the input word in the corpus
    int five = 0;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      
      if (b == -1)
        five++;
    }
    if(five == cn){
      fprintf(f_for_acc, "%s, ", spid);

      char tmp500[50];
      f_top500 = fopen("../top500track", "r");
      if (f_top500 == NULL) {
        printf("top500 file not found\n");
        return -1;
      }

      printf("top500\n");

      for(i = 0; i < 499; i++){
        fscanf(f_top500, "%s", tmp500);
        fprintf(f_for_acc, "spotify:track:%s, ", tmp500);
      }
      fscanf(f_top500, "%s", tmp500);
      fprintf(f_for_acc, "spotify:track:%s\n", tmp500);
      five_not_match++;
      fclose(f_top500);
      continue;
    }

    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[input_num][a] = -1;
    for (a = 0; a < N; a++) bestw[input_num][a][0] = 0;
    int tmp = 0;
    for (c = 0; c < words; c++) {
      if(strlen(&vocab[c * max_w]) != 22)
        continue;
      tmp++;
      a = 0;  
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;//the same word as input word
      if (a == 1) continue;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
      for (a = 0; a < N; a++) {
        if (dist > bestd[input_num][a]) {
          for (d = N - 1; d > a; d--) {
            bestd[input_num][d] = bestd[input_num][d - 1];
            strcpy(bestw[input_num][d], bestw[input_num][d - 1]);
          }
          bestd[input_num][a] = dist;
          strcpy(bestw[input_num][a], &vocab[c * max_w]);
          break;
        }
      }
    }
    fprintf(f_for_acc, "%s, ", spid);
    for(i = 0; i < N; i++){
      if(i == N - 1)
        fprintf(f_for_acc, "spotify:track:%s\n", bestw[input_num][i]);
      else
        fprintf(f_for_acc, "spotify:track:%s, ", bestw[input_num][i]);

    }
  }

  printf("finish, gc:%d\n", gc);
  printf("five_not_match: %d\n", five_not_match);
  fclose(f_test);
  fclose(f_for_acc);
  
  return 0;
}
 