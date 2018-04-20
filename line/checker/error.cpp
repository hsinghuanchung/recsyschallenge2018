#include"retrive.h"
#include<cstdio>
#include<cstring>

int main(int argc,char *argv[])
{
	int size = atoi(argv[1]);

	mapping dict(argv[1],argv[2],atoi(argv[3]));
	
	printf("test\n");
	dict.goal(argv[4],20000);
	

}
