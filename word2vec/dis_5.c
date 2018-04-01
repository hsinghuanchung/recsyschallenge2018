/*

gcc dis_5.c -o dis_5

./dis_5 vectors.bin [input file name]         [output file name]
./dis_5 vectors.bin five_input_800000_1000000 dis5_ans_800000_1000000

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
  char st1[max_size];
  char *bestw[10][N];
  char file_name[max_size], st[100][max_size], file_name2[max_size], file_name3[max_size];
  float dist, len, bestd[10][N], vec[max_size];
  long long words, size, a, b, c, d, cn, bi[100];
  char ch;
  float *M;
  char *vocab;


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

  int input_num = 0, end_flag = 0;
  char tmp_s[max_w];

  f_for_acc = fopen(file_name3, "w");

  while (1) {
    for (a = 0; a < N; a++) bestd[input_num][a] = 0;
    for (a = 0; a < N; a++) bestw[input_num][a][0] = 0;
    //printf("Enter %d word: ", input_num+1);
    //printf("Enter word or sentence (EXIT to break): ");
    a = 0; 
    fscanf(f_test, "%s", st[a]);
    while(1){
      if(strcmp(st[a], "ENDFILE") == 0){
        end_flag = 1;
        break;
      }
      else if(strcmp(st[a], "NEXT") == 0)
        break;
      a++;
      if(fscanf(f_test, "%s", st[a]) != EOF)
        continue;
      else
        break;
    }
    cn = a;
   
    /*while (1) {
      //st1[a] = fgetc(f_test);

      //fscanf(f_test, "%s", st1[a]);
      st1[a] = fgetc(stdin);
      //printf("%c\n", st1[a]);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        //end_flag = 1;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;*/

    //count the position of the input word in the corpus
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      //printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        //printf("Out of dictionary word!\n");
        //break;
      }
    }
    if (b == -1) continue;
    //printf("\n                                         track_uri       Cosine distance\n------------------------------------------------------------------------\n");
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
    for (c = 0; c < words; c++) {
      a = 0;  
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
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
    //for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[input_num][a], bestd[input_num][a]);
    
    for(i = 0; i < N; i++){
      //fprintf(f_for_acc, "%s ", aver[i].aword);
      fprintf(f_for_acc, "%s ", bestw[input_num][i]);
    }
    //input_num++;

    if(end_flag){
      fprintf(f_for_acc, "ENDFILE");
      break;
    }
    else
      fprintf(f_for_acc, "NEXT ");
  }

  
  /*
  //////This is used in my own way of calculating the weight of five distance//////
  //calculate average weight for each uri
  int j, count = 0;
  for(i = 0; i < input_num; i++){
    for(j = 0; j < N; j++){
      if(i == 0){//bestw[0][j], copy directly
        strcpy(aver[count].aword, bestw[i][j]);
        aver[count].adist = bestd[i][j];
        aver[count].cnt = 1;
        count++;
      }
      else{
        int k, flag = 0;
        for(k = 0; k < count; k++){
          if(strcmp(aver[k].aword, bestw[i][j]) == 0){
            //printf("hiiiiiiiiiiiiiiiiii, i: %d, j: %d\n", i, j);  
            aver[k].adist += bestd[i][j];
            aver[k].cnt ++;
            flag = 1;
            break;
          }
        }
        if(flag == 0){//not found, create one
          strcpy(aver[count].aword, bestw[i][j]);
          aver[count].adist = bestd[i][j];
          aver[count].cnt = 1;
          count++;
        }
      }
    }
  }
  //calculate final average distance
  int total_weight = N * input_num;
  for(i = 0; i < count; i++){
    aver[i].final_dist = (aver[i].adist / aver[i].cnt);//simple, have to be modified
    //aver[i].final_dist = (aver[i].adist / total_weight);
  }

  //sort by final_dist
  qsort(aver, count, sizeof(aver[0]), cmpfunc);

  //print final output
  printf("count: %d\n", count);
  for(i = 0; i < count; i++)
    printf("%s, %f, %f\n", aver[i].aword, aver[i].final_dist, aver[i].cnt);
  
  printf("\nFinal cosine distance:\nTotal weight: %d\n", total_weight);
  printf("\n                                         track_uri         Simple weight\n------------------------------------------------------------------------\n");
  for (i = 0; i < 10; i++) 
    printf("%50s\t\t%f\n", aver[i].aword, aver[i].final_dist);
  */

  
  printf("finish\n");

  fclose(f_test);
  fclose(f_for_acc);
  return 0;
}
 
