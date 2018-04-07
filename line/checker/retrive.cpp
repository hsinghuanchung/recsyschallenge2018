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
void mapping::goal()
{
	int rc;
	int ss=arr.size()/this->thread_num;
	pthread_t thread[this->thread_num];
	pair<mapping*,int> parm[this->thread_num];
	void *res;

	for(int i=0;i<this->thread_num;i++)
	{
		this->pt[i].first = i*ss;
		this->pt[i].second =(i+1)*ss;

		if( i == (this->thread_num-1))
			this->pt[i].second = arr.size();

		thread[i] = i;

		parm[i].first = this;
		parm[i].second = i;

		rc = pthread_create(&thread[i],NULL,mapping::find,(void*)&parm[i]);
	}
	for(int i=0;i<this->thread_num;i++)
		rc = pthread_join(thread[i],&res);
}
void* mapping::find(void *parm)
{
	pair<mapping*,int> *temp = (pair<mapping*,int>*) parm;
	priority_queue< temp_prior > tt;
	
	double qq;
	temp_prior h; 
	int i,j,ss;
	char file[128]="ans";
	sprintf(file,"ans_%d_[%d_%d].out",temp->first->size,temp->first->pt[temp->second].first,temp->first->pt[temp->second].second);

	ofstream ofs(file);

	for(i=temp->first->pt[temp->second].first ; i<temp->first->pt[temp->second].second ; i++)
	{
		for( j=0,ss = temp->first->arr.size() ; j<ss ; j++)
		{
			if(i == j)
				continue;
			qq = 0;

			for(int k=0 ; k< temp->first->size ; k++)
				qq += (temp->first->arr[i][k] - temp->first->arr[j][k]) * (temp->first->arr[i][k] - temp->first->arr[j][k]);
			h.value = sqrt(qq);
			h.node = j;
			
			tt.push(h);
			if(tt.size() > 40)
				tt.pop();

		}
		for( j=0 ; j<40 ; j++)
		{
			h = tt.top();
			tt.pop();
			
			ofs<<h.value<<" "<<h.node<<endl;
		}
	}
}

