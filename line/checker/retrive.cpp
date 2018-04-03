#include"retrive.h"
using namespace std;

mapping::mapping()
{
	mapping("./../../data/mapping","./../../data/out_result",64);
}
mapping::mapping(const char *file_name,const char *file,int size)
{
	//open the file here and make it into file
	this->size = size;



	ifstream in(file_name); 
	string temp;
	int index,num;
	
	printf("%s\n",file_name);

	getline(in,temp);
	getline(in,temp);

	while(getline(in,temp))
	{
		//cout<<index<<endl;
		index = temp.rfind(' ');
				
		num = atoi(&temp.c_str()[index+1]);	
	    temp.erase(temp.begin()+index,temp.end());

		if( dic.find(temp) == dic.end() )
		{
			dic[temp] = num;
		}
		else
		{
			cout<<"it seems to be some error\n";	
		}
		re.push_back(temp);
	}

	in.close();

	string::size_type s1,s2;
	
	double val;
	char size_num[128];
	sprintf(size_num,"%s%d",file,size);
	
		
	in.open(size_num);
	vector<double> data;
	
	arr.clear();
	while(getline(in,temp))
	{
		data.clear();
		index = stoi(temp,&s1);
		for(int i=0 ; i<this->size ; i++)
		{
			val = stod(temp.substr(s1),&s2);
			s1 += s2;  
			data.push_back(val);
		}
		arr.push_back(data);
	}
}
int  mapping::query_int(string arr)
{
	return this->dic[arr];
}
string mapping::query_url(int index)
{
	return this->re[index];
}

void mapping::find()
{
	priority_queue< temp > tt;
	
	double qq;
	temp h; 

	char file[128]="ans";
	sprintf(file,"ans_%d.out",this->size);

	ofstream ofs(file);

	for(int i=0 ; i<arr.size() ; i++)
	{
		for(int j=0 ; j<arr.size() ; j++)
		{
			if(i == j)
				continue;
			qq = 0;

			for(int k=0 ; k< this->size ; k++)
				qq += (arr[i][k]-arr[j][k]) * (arr[i][k]-arr[j][k]);
			h.value = sqrt(qq);
			h.node = j;
			
			tt.push(h);
			if(tt.size() > 40)
				tt.pop();

		}
		for(int j=0 ; j<40 ; j++)
		{
			h = tt.top();
			tt.pop();
			
			ofs<<h.value<<" "<<h.node<<endl;
		}
	}
}

