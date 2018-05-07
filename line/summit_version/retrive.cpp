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
	size_t pt,now;
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
				dic[temp].first = num;
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
					else if(temp[i]==' ' && key_word.size())
					{
						i++;
						break;
					}		
					i++;
					if(i==temp.size())
						break;
				}

				if(key_word[ key_word.size()-1 ] == 's' && key_word[ key_word.size()-2 ] == '\'' )
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
					number = key_word+" "+to_string(count);	
					
					if( dic.find(number) == dic.end() )
					{
						dic[number].first = num;
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

	count=0;
	in.open("/mnt/data/recsys_spotify/line_data/song_information/set");
	while(getline(in,temp))
	{
		now = pt = 0;
		pt = temp.find(' ',now);
		dic[temp.substr(now,pt-now)].second = count;
		if(pt == now || pt == string::npos)
			break;
		count++;
		
		now = pt+1;
		while(temp[now]==' ')
			now++;
	}
	in.close();

	pop_song.clear();
	in.open("/mnt/data/recsys_spotify/line_data/song_information/song_sort");
	do
	{
		getline(in,temp);
		now = pt = 0;
		pt = temp.find('_',now);

		pop_song.push_back(temp.substr(now,pt-now));
	}while(pop_song.size()<500);
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
	if(dic.find(arr) == dic.end())
		return -1;

	return this->dic[arr].first;
}
string mapping::query_url(int index)
{
	return this->re[index];
}
int  mapping::query_set(string arr)
{
	if(dic.find(arr) == dic.end())
		return -1;

	return this->dic[arr].second;
}
void mapping::goal(char* in_file,int range=-1)	//input the query file here
{
	int index,i,j,num,count=0;
	size_t pt,now;

	ifstream in(in_file); 
	string temp;
	vector<int> qu;
	vector<double> mean_vec(this->size,0),mean_temp;


	this->mean_vector.clear();

	fill(this->class_set.begin(),this->class_set.end(),-1);

	getline(in,temp);
	while(in)
	{
		fill(mean_vec.begin(),mean_vec.end(),0);
		//first deal with title
		//for the same word there might be many version, we simply took its average

		pt = now = num = 0;
		printf("%d\n",num);
		while(1)
		{
			fill(mean_temp.begin(),mean_temp.end(),0);

			pt = temp.find(" ",now);	
			if(pt == string::npos)
				break;
			
			this->find_average(temp.substr(now,pt-now),mean_temp);
			
			now = pt+1;
			while(temp[now]==' ' && now<temp.size())
				now++;

			for(i=0;i<mean_temp.size();i++)
				if(mean_temp[i]!=0)
					break;
			
			if(i==mean_temp.size())
				continue;

			for(i=0;i<mean_temp.size();i++)
				mean_vec[i] += mean_temp[i];
			num ++;
		}
		printf("%d\n",num);

		//second deal with the track
		getline(in,temp);
		pt = now = 0;
		while(1)
		{

			pt = temp.find(" ",now);

			index = this->query_int(temp.substr(now,pt-now));	
			
			if(index == -1)
				continue;
			
			if(pt == string::npos)
				break;
			
			now = pt+1;
			while(temp[now]==' ')
				now++;

			class_set[count] = this->query_set(temp.substr(now,pt-now));

			num++;
			for(j=0 ; j< this->size ; j++)
				mean_vec[j] += this->arr[index][j];
		}
		printf("%d\n",num);

		if(num!=0)
			for(i=0;i<this->size;i++) 
				mean_vec[j] /= num;

		this->mean_vector.push_back(mean_vec);
		count ++;
		printf("%d\n",count);
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

	double one;

	temp_prior h; 
	long long i,j;
	int k;

	char file[128];
	sprintf(file,"/mnt/data/recsys_spotify/line_data/qans_challenge_%d_[%lld_%lld].out",temp->first->size,temp->first->pt[temp->second].first,temp->first->pt[temp->second].second);
	FILE *out_file = fopen(file,"w");

	priority_queue< temp_prior >tt;
	vector< temp_prior > rever;
	vector<string> left;
	queue<int> ans;

	for(i=temp->first->pt[temp->second].first ; i<temp->first->pt[temp->second].second ; i++)
	{
		for(j=0;j<temp->first->size ; j++)
			if(temp->first->mean_vector[i][j] == 0)
				break;

		if(j==temp->first->size)
		{
			///out_put first 500 pop
			for( j=0 ; j<500 ; j++)
				fprintf(out_file,"%s ",temp->first->pop_song[j].c_str());
			fprintf(out_file,"\n");
		}
		else
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
					one += (temp->first->mean_vector[i][k] - temp->first->arr[j][k])*(temp->first->mean_vector[i][k] - temp->first->arr[j][k]);

				h.node = j;
				h.value = one;

				tt.push(h);
				if(tt.size()>1000)
					tt.pop();

			}

			rever.clear();
			for( j=0 ; j<1000 ; j++)
			{
				h = tt.top();
				tt.pop();
				rever.push_back(h);
			}
			
			//use the set group result to double fine check the output
			left.clear();
			k=500;
			for( j=999 ; j>=0 ; j--)
			{
				url = temp->first->query_url(rever[j].node);
				if( temp->first->query_set(url) == temp->first->class_set[i] )
				{
					fprintf(out_file,"%s ",url.c_str());
					k--;
				}		
				else
					left.push_back(url);
			}	
			for(j=0;j<k;j++)
				fprintf(out_file,"%s ",left[j].c_str());

			fprintf(out_file,"\n");
		}
	}
	fclose(out_file);
	return 0;
}

void mapping::find_average(string title,vector<double> &mean_temp)
{
	//in this part we query every version of the word and find for its average
	string title_temp;
	int i,j,index;
	cout<<title<<"-"<<endl;


	for(i=0 ; ;i++)
	{
		title_temp = title + " " + to_string(i);
		index = this->query_int(title_temp);

		if(index == -1)
			break;

		for(j=0;j<this->size;j++)
			mean_temp[j] += this->arr[index][j];
	}
	if(i == 0)
		return ;

	for(j=0 ; j<this->size ; j++)
		mean_temp[j] /= i;
}
