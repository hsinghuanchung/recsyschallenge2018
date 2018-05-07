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
	string key_word;

	ifstream in(file_name); 
	string temp;
	int index,num,i,count;
	size_t pt;
	string number;


	printf("%s\n",file_name);

	getline(in,temp);
	getline(in,temp);

	while(getline(in,temp))
	{
		//cout<<index<<endl;
		index = temp.rfind(' ');
				
		num = atoi(&temp.c_str()[index+1]);	
	    temp.erase(temp.begin()+index,temp.end());

		re.push_back(temp);

		//check if it is title or track


		if(temp.find("spotify:track:"))
		{
			if( dic.find(temp) == dic.end() )
				dic[temp] = num;
			else
				cout<<"it seems to be some error\n";	
		}
		else
		{
			i=0;
			while(1)
			{
				key_word = "";
				while(1)
				{
					if('0'<=temp[i] && temp[i]<='9')
						key_word += temp[i];
					else if('a'<=temp[i] && temp[i]<='z')
						key_word += temp[i];
					else if('A'<=temp[i] && temp[i]<='Z')
						key_word += temp[i]-32;
					else if(temp[i]=='_')
						key_word += temp[i];
					else if(temp[i]=='\'')
						key_word += temp[i];
					else if(temp[i]==' ')
					{
						i++;
						break;
					}		
					i++;
					if(i==temp.size())
						break;
				}

				if(key_word[ key_word.size()-1 ] == 's' && key_word[ key_word.size()-1 ] == '\'' )
					key_word.erase(key_word.end()-2,key_word.end());

				if(i==temp.size())//last one11
				{
					pt = key_word.rfind('_');
					if(pt == 0 )
						break;
					key_word = key_word.substr(0,pt);
				}	

				count = 0;
				while(1)
				{	
					pt = key_word.rfind(' ');
					number = key_word+" "+to_string(count);	
					
					if( dic.find(number) == dic.end() )
					{
						dic[number] = num;
						break;
					}
					else
					{
						count++;	
					}	
				}
				if(i==temp.size())
					break;	
			}
		}
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
	this->testing();
}
int  mapping::query_int(string arr)
{
	if(dic.find(arr) == dic.end())
		return -1;

	return this->dic[arr];
}
string mapping::query_url(int index)
{
	return this->re[index];
}

void mapping::goal(char* in_file,int range=-1)	//input the query file here
{
	int now=0,index,pt,i,j,num;
	size_t pt,now;

	ifstream in(in_file); 
	string temp;
	vector<int> qu;
	vector<double> mean_vec(this->size,0),mean_temp;


	this->mean_vector.clear();

	getline(in,temp);
	while(in)
	{
		fill(mean_vec.begin(),mean_vec.end(),0);
		//first deal with title
		//for the same word there might be many version, we simply took its average

		pt = now = num = 0;
		while(1)
		{
			pt = temp.find(" ",now);	
			mean_temp = this->find_average(temp.substr(now,pt-now));

			for(i=0;i<mean_temp.size();i++)
				mean_vec[i] += mean_temp[i];

			num += 1;
			if(pt == -1)
				break;
			now = pt+1;
		}

		//second deal with the track
		getline(in,temp);
		now=0;
		qu.clear();
		while(1)
		{
			pt = temp.find(" ",now);
			index = this->query_int(temp.substr(now,pt-now));	

			if(index == -1)
				continue;
			
			num++;
			
			for(j=0 ; j< this->size ; k++)
				mean_vec[j] += this->arr[index][j];

			now = pt+1;
		}

		for(i=0;i<this->size;i++) 
			mean_vec[j] /= num;

		this->mean_vector.push_back(mean_vec);
	}

	int ss = this->mean_vector.size()/this->thread_num;

	pthread_t thread[this->thread_num];
	pair<mapping*,int> parm[this->thread_num];
	void *res;

	for(int i=0;i<this->thread_num;i++)
	{
		this->pt[i].first = i*ss;
		this->pt[i].second =(i+1)*ss;

		if( i == (this->thread_num-1))
			this->pt[i].second = this->mean_vector.size();

		thread[i] = i;

		parm[i].first = this;
		parm[i].second = i;

		pthread_create(&thread[i],NULL,mapping::find,(void*)&parm[i]);
	}
	for(int i=0;i<this->thread_num;i++)
		pthread_join(thread[i],&res);
}

void* mapping::find(void *parm)
{
	pair<mapping*,int> *temp = (pair<mapping*,int>*) parm;
	string url;

	double one,all;

	temp_prior h; 
	long long i,j,l;
	int k;

	char file[128];
	sprintf(file,"/mnt/data/recsys_spotify/line_data/ans_challenge_%d_[%lld_%lld].out",temp->first->size,temp->first->pt[temp->second].first,temp->first->pt[temp->second].second);
	FILE *out_file = fopen(file,"w");

	priority_queue< temp_prior >tt;
	vector< temp_prior > rever;


	for(i=temp->first->pt[temp->second].first ; i<temp->first->pt[temp->second].second ; i++)
	{
		while(!tt.empty())
			tt.pop();
		
		for(j=0 ; j<temp->first->arr.size() ; j++)
		{
			url = temp->first->query_url(j);
			if(url.find("spotify:track:") == std::string::npos)	//this is the title
				continue;

			one = 0;
			for(k=0 ; k< temp->first->size ; k++)
				one += (mean_vec[k] - temp->first->arr[j][k])*(mean_vec[k] - temp->first->arr[j][k]);

			h.node = j;
			h.value = one;

			tt.push(h);
			if(tt.size()>600)
				tt.pop();

		}

		rever.clear();
		for( j=0 ; j<600 ; j++)
		{
			h = tt.top();
			tt.pop();
			rever.push_back(h);
		}
		
		//use the set group result to double fine check the output

		for( j=499 ; j>=0 ; j--)
			fprintf(out_file,"%lld ",rever[j].node);
		fprintf(out_file,"\n");

	}
	fclose(out_file);
	return 0;
}

vector<double> mapping::find_average(string title)
{
	//in this part we query every version of the word and find for its average
	string title_temp;
	int i,j,index;
	vector<double> mean_temp(this->size,0);

	for(i=0 ; ;i++)
	{
		title_temp = title + " " + to_string(i);
		index = this->query_int(title_temp);

		if(index == -1)
			break;
			
		for(j=0;j<this->size;j++)
			mean_temp[j] += this->arr[index][j];
	}
	for(i=0 ; i<this->size ; i++)
		mean_temp[i] /= i;
	
	return mean_temp;
}