
#https://www.readwithu.com/Article/PythonBasis/python5/Cycle.html
#%%
#-*-coding:utf-8-*-
#-----------------------list的使用----------------------------------

# 1.一个产品，需要列出产品的用户，这时候就可以使用一个 list 来表示
user=['liangdianshui','twowater','两点水']
print('1.产品用户')
print(user)

# 2.如果需要统计有多少个用户，这时候 len() 函数可以获的 list 里元素的个数
len(user)
print('\n2.统计有多少个用户')
print(len(user))

# 3.此时，如果需要知道具体的用户呢？可以用过索引来访问 list 中每一个位置的元素，索引是0从开始的
print('\n3.查看具体的用户')
print(user[0]+','+user[1]+','+user[2])

# 4.突然来了一个新的用户，这时我们需要在原有的 list 末尾加一个用户
user.append('茵茵')
print('\n4.在末尾添加新用户')
print(user)

# 5.又新增了一个用户，可是这个用户是 VIP 级别的学生，需要放在第一位，可以通过 insert 方法插入到指定的位置
# 注意：插入数据的时候注意是否越界，索引不能超过 len(user)-1
user.insert(0,'VIP用户')
print('\n5.指定位置添加用户')
print(user)

# 6.突然发现之前弄错了，“茵茵”就是'VIP用户'，因此，需要删除“茵茵”；pop() 删除 list 末尾的元素
user.pop()
print('\n6.删除末尾用户')
print(user)

# 7.过了一段时间，用户“liangdianshui”不玩这个产品，删除了账号
# 因此需要要删除指定位置的元素，用pop(i)方法，其中i是索引位置
user.pop(1)
print('\n7.删除指定位置的list元素')
print(user)

# 8.用户“两点水”想修改自己的昵称了
user[2]='三点水'
print('\n8.把某个元素替换成别的元素')
print(user)

# 9.单单保存用户昵称好像不够好，最好把账号也放进去
# 这里账号是整数类型，跟昵称的字符串类型不同，不过 list 里面的元素的数据类型是可以不同的
# 而且 list 元素也可以是另一个 list
newUser=[['VIP用户',11111],['twowater',22222],['三点水',33333]]
print('\n9.不同元素类型的list数据')
print(newUser)

#%%
dict1={'liangdianshui':'111111' ,'twowater':'222222' ,'两点水':'333333'}
print(dict1)

#del dict element
#del dict1['liangdianshui']


# %%
#-*-coding:utf-8-*-
dict1={'liangdianshui':'111111' ,'twowater':'222222' ,'两点水':'333333','twowater':'444444'}
print(dict1)
print(dict1['twowater'])
#(1) dict （字典）是不允许一个键创建两次的，但是在创建 dict （字典）的时候如果出现了一个键值赋予了两次，会以最后一次赋予的值为准

# %%
set1=set([123,456,789])
print(set1)

# %%

set1=set('hello')
set2=set(['p','y','y','h','o','n'])
print(set1)
print(set2)

# 交集 (求两个 set 集合中相同的元素)
set3=set1 & set2
print('\n交集 set3:')
print(set3)
# 并集 （合并两个 set 集合的元素并去除重复的值）
set4=set1 | set2
print('\n并集 set4:')
print(set4)
# 差集
set5=set1 - set2
set6=set2 - set1
print('\n差集 set5:')
print(set5)
print('\n差集 set6:')
print( set6)


# 去除海量列表里重复元素，用 hash 来解决也行，只不过感觉在性能上不是很高，用 set 解决还是很不错的
list1 = [111,222,333,444,111,222,333,444,555,666]  
set7=set(list1)
print('\n去除列表里重复元素 set7:')
print(set7)


# %%
# 1--100 cumsum

num=0
tim=1
while tim<=100:
    num=num+tim
    tim+=1
    



print(num)
#%%
for i in range(0, 10):
    print(i)# %%


# %%

i = 0
while i < 10:
    print(i)
    i = i + 1

# %%#
#统计 1 到 100 之间的奇数和
summ=0
num1=1

for num1 in range(101):
    if num1%2!=0:
        summ=summ+num1

#use while loop to finish

sum2=0
num2=1

while num2<=100:
    if num2%2==0:
        num2+=1
        continue
    sum2=sum2+num2
    num2+=1

print(sum2)
        







# %%
# the input function is going to return a str
# you need to convert the dtype into the one you want
result=input('please entry value: ')
result2=int(input('please entry value: '))





# %%
#validating user input
def user_input_validation():

    #initial
    check='wrong'
    ok_range= range(1,11)
    within_range=False

    while check.isdigit()==False or within_range==False:
        check=input('give me a rating(1--10):')

        #check is the number digit
        if check.isdigit()==False:
            print('Sorry this is not a digit!!!!!')
            
        #check is the number within range
        if check.isdigit()==True:
            if int(check) in ok_range:
                within_range=True
            else:
                within_range=False
                print('out of range')

    return int(check)
#%%


user_input_validation()


# %%
# let me try to finish this function by myself

def User_num_validation():
    value='Wrong'

    desired_range=range(11)

    within_range=False



    while value.isdigit()==False or within_range==False:

        value=input('Give me a rating bb:')
        

        #notice value is not digit
        if value.isdigit()==False:
            print('value is not digit')
        # when num in range break thee loop
        # when num out of the range, continue the range and give notice
        else:
            if int(value) in desired_range:
                #within_range=True
                break
            else:
                within_range=False
                print('number out of range')
    return int(value)
        
''' Summary for this while loop function:

while condition1 or condition2:---->give of condition, if condition safistied, the loop will continue
# in order to let the loop run for the very first time, we need to give conditions a 
#initial value.

while the loop running, we add more if condition used to give out error notive
and if the value meet our condition, set the predefined parameter to True
so that break the while condition!!

'''






# %%
User_num_validation()


3
# %%
''' create tic toc toe game'''


game_list=[0,1,2]
def display_game(game_list):
    print('here is the current list: ')
    print(game_list)

# %%
display_game(game_list)
#%%
def position_choice():
    choice='Wrong'
    while choice not in ['0','1','2']:
        choice=input('Pick a position (0,1,2): ')

        if choice not in ['0','1','2']:
            print('Sorry, Invalid choice!')
    return int(choice)

#%%

position_choice()




# %%
#'''replacement'''

def replacement_choice(game_list,position):
    user_placement=input('Type a string to place at position: ')
    game_list[position]=user_placement
    return game_list

#%%

replacement_choice(game_list,1)
#%%
def position_choice():
    choice='Wrong'
    while choice not in ['Y','N']:
        choice=input('wanna keep play?:  ')

        if choice not in ['Y','N']:
            print('Sorry, i dont understand! please choose Y or N')
    if choice=='Y':
        return True
    else:
        return False













# %%
x='apple'
assert x == "apeple"


# %%
a=[1,2,3,4,5,6,7,8,9,10]
print(np.std(a))

a1=6
b=15
c=34

print(np.std([a1,c,b]))

# %%
'''coin exchange week3 question 1'''

def coin_changre(m):
    option=[10,5,1]
    move=0
    for face in range(len(option)):
        move+= m//option[face]
        m=m%option[face]
    return move


# %%
'''fractional kanpsack week3 question 2'''


def get_optimal_value(capacity, values, weights):

    #transform list to array: easier to calculate ratio
    weights=np.array(weights)
    values=np.array(values)
    ratio=values/weights
    
    # take_from: trace how much we take from xxx resource
    # total_take : record total stolen value used to return at the end
    take_from=0
    total_take=0



    while capacity>0:


        max_value=ratio.max() #find the best item
        max_index=ratio.argmax() #find the best item's index
        max_value_amount=weights[max_index] # find the best item's total amount

        take_from=min(max_value_amount,capacity)

        total_take=total_take+take_from*max_value

        #update weight array and ratio array

        weights=np.delete(weights,max_index)
        ratio=np.delete(ratio,max_index)
        values=np.delete(values,max_index)


        capacity-=take_from

    return total_take
    

# %%
get_optimal_value(50, values= [60,100,120],weights=[20,50,30])

# %%
get_optimal_value(10, values=[500],weights=[30])


# %%
def fractional_knapsack(value, weight, capacity):

    """Return maximum value of items and their fractional amounts.
 
    (max_value, fractions) is returned where max_value is the maximum value of
    items with total weight not more than capacity.
    fractions is a list where fractions[i] is the fraction that should be taken
    of item i, where 0 <= i < total number of items.
 
    value[i] is the value of item i and weight[i] is the weight of item i
    for 0 <= i < n where n is the number of items.
 
    capacity is the maximum weight.
    """
    # index = [0, 1, 2, ..., n - 1] for n items
    index = list(range(len(value)))
    # contains ratios of values to weight
    ratio = [v/w for v, w in zip(value, weight)]
    # index is sorted according to value-to-weight ratio in decreasing order
    index.sort(key=lambda i: ratio[i], reverse=True)
 
    max_value = 0
    fractions = [0]*len(value)
    for i in index:
        if weight[i] <= capacity:
            fractions[i] = 1
            max_value += value[i]
            capacity -= weight[i]
        else:
            fractions[i] = capacity/weight[i]
            max_value += value[i]*capacity/weight[i]
            break
 
    return max_value, fractions
#%%
fractional_knapsack([60,100,120],[20,50,30], 50)


# %%















#%%


def fractional_knapsack2(value, weight, capacity):
    index=list(range(len(value)))
    ratio=[ v/w for v,w in zip(value,weight)]
    index.sort(key=lambda i:ratio[i], reverse=True)

    max_value=0

    for i in index:
        if weight[i]<=capacity:
            max_value+=value[i]
            capacity-=weight[i]
        else:

            max_value+=capacity*ratio[i]
            break
    return max_value


# %%
fractional_knapsack2(value=[60,100,120], weight=[20,50,30], capacity=50)


# %%
fractional_knapsack2(value=[500], weight=[30], capacity=10)


# %%
'''class and object'''

class Animal:
    def __init__(self,color,weight,kind):
        self.color=color
        self.weight=weight
        self.kind=kind
        print('init')
    
    def rank(self,top=2):
        self.top=top
        print(self.top,self.color,self.weight,self.kind)

# %%
zhu=Animal('yellow','500kg','pig')
zhu.rank(1099999)

 # %%
def max_num(*args):
    return max(args)


# %%
a,b,c=map(int,input().split())
max_num(a,b,c)

# %%
mylist=[1,2,3,4,5,6,7,8,9]
for i in mylist:
    if i==5:
        continue
    print(i, end=' ')



# %%
def swap(a,b):
    a,b=b,a
    return a,b

# %%
a=1
b=99
a,b=swap(a,b)

# %%
class Solution:

    def swapIntegers(self,A, index1, index2):
        A[index1],A[index2]=A[index2],A[index1]
        #在这里并不需要return A, 由于list给定的是具体的address，改变直接就完成了
        #但在单一参数的时候（variable）必须要return 并且重新赋值

# %%
A=[1,2,3,4,5]
swapIntegers(A,index1= 0, index2=4)


# %%


# %%

class Student:
    def __init__(self, id):
        self.id = id

class Class:

    '''
     * Declare a constructor with a parameter n which is the total number of
     * students in the *class*. The constructor should create n Student
     * instances and initialized with student id from 0 ~ n-1
    '''
    # write your code here
    def __init__(self, n):
        self.students = []
        for i in range(n): 
            self.students.append(Student(i))
        
# %%
cls=Class(5)
# %%
print(cls.students)

# %%

class Person:
    def __init__(self, name, age):
        self.name = name
        self.__age = age
        self.speak()
    def speak(self):
        print(self.name, self.__age)
#%%

p1 = Person("John", 36)
#%%
p1._Person__age=99999

#%%
p1.speak()


#%%
print(p1.name)
print(p1.age) 

# %%
class Student:
    def __init__(self,id):
        self.id=id

class Class:
    def __init__(self,n):
        self.students=[]
        for i in range(n):
            self.students.append(Student(i))

# %%
team2=Class(6)

# %%
team2.students

# %%
class Student:
    def __init__(self, id):
        self.id = id

class Class:

    '''
     * Declare a constructor with a parameter n which is the total number of
     * students in the *class*. The constructor should create n Student
     * instances and initialized with student id from 0 ~ n-1
    '''
    # write your code here
    def __init__(self, n):
        self.students=[]
        for i in range(n):
            self.students.append(Student(i))


# %%
class Student:
    def __init__(self, name):
        self.name = name
    
    def set_score(self):
        self.score = 90
    
stu = Student('Jack')
print(stu.name, stu.score)
stu.set_score()
print(stu.name, stu.score)

# %%
class ArrayListManager:
    '''
     * @param n: You should generate an array list of n elements.
     * @return: The array list your just created.
    '''
    def create(self, n):
        # Write your code here
        return [i for i in range(n)]
    
    
    '''
     * @param list: The list you need to clone
     * @return: A deep copyed array list from the given list
    '''
    def clone(self, list):
        
        return [i for i in list]
    
    
    
    '''
     * @param list: The array list to find the kth element
     * @param k: Find the kth element
     * @return: The kth element
    '''
    def get(self, list, k):
        # Write your code here
        
        return list[k]
    
    
    '''
     * @param list: The array list
     * @param k: Find the kth element, set it to val
     * @param val: Find the kth element, set it to val
    '''
    def set(self, list, k, val):
        # write your code here
        list[k]=val
        
    
    
    '''
     * @param list: The array list to remove the kth element
     * @param k: Remove the kth element
    '''
    def remove(self, list, k):
        
        # write tour code here
        
        list.pop(k)
    
    
    '''
     * @param list: The array list.
     * @param val: Get the index of the first element that equals to val
     * @return: Return the index of that element
    '''
    def indexOf(self, list, val):
        if val in list:
            
            return list.index(val)
        else:
            return -1
        
        # Write your code here




#%%
from queue import Queue

que=Queue(maxsize=100)

for i in range(20):
    que.put(i)

print(que.qsize())

# %%
que.get(0)

# %%
stack=[]

# %%
class Solution:
    """
    @param s: the given string
    @return: whether this string is valid
    """
    def checkValidString(self, s):
        # Write your code here
        stack=[]
        for i in s:
            if i=='(' or i=='*':
                stack.append(i)
            else:
                if not stack:
                    return False
                if i=='(' and stack[-1]!=')' or i=='*' and stack[-1]!=')':
                    return False
            stack.pop()
        return not stack
                    


# %%
def checkValidString( s):
    stack=[]
    for i in s:
        if i=='(':
            stack.append(i)
        else:
            if not stack:
                return False
            if i=='(' and stack[-1]!=')':
                return False
            stack.pop()
    return stack

# %%
def checkValidString(s):
    stack=[]
    for ch in s:
        if ch=='{' or ch=='['or ch=='(':
            stack.append(ch)
        else:
            if not stack:
                return False
            if ch==']' and stack[-1]!='[' or ch=='{' and stack[-1]!='{' or ch=='(' and stack[-1]!=')':
                return False
            stack.pop()
    return not stack
            

# %%
    def rotateString( s, offset):
        # write your code here
        
        
        s2=s[-offset:]
        
        return s2
# %%
    
def rotateString( s, offset):
        # write your code here
        
        
        s2=s[-offset:]
        
        s1=s[:-offset]
        
        return s2+s1

# %%
def flatten(n):

    for i in n:
        if isinstance(i,int):
            mylst.append(i)
        elif isinstance(i,list):
            for ii in i:


#%%

df=pd.read_csv('C:/Users/Mr.Goldss/Desktop/电力专题--样例数据/20180102.txt',sep='\t',
encoding= 'unicode_escape')

# %%
import pandas as pd
df2=pd.read_csv('C:/Users/Mr.Goldss/Desktop/电力专题--样例数据/20180102.txt',sep='\t',encoding='GBK')
df2.head()

# %%
class Student:
    def __init__(self, id):
        self.id = id;

class Class:

    '''
     * Declare a constructor with a parameter n which is the total number of
     * students in the *class*. The constructor should create n Student
     * instances and initialized with student id from 0 ~ n-1
    '''
    # write your code here
    def __init__(self,n):
        self.students=[Student(i) for i in range(n)]
    
    def s(self):
        return self.students


# %%
c=Class(10)




# %%

        
class Solution:
    # @param {int[]} nums an array of integers
    # @return {int} the number of unique integers
    def deduplication( nums):
        # Write your code here
        n = len(nums)
        if n == 0:
            return 0
            
        nums.sort()
        result = 1
        for i in range(1, n):
            if nums[i - 1] != nums[i]:
                nums[result] = nums[i]
                result += 1
                
        return result

# %%
def deduplication( nums):
        # Write your code here
        n = len(nums)
        if n == 0:
            return 0
            
        nums.sort()
        result = 1
        for i in range(1, n):
            if nums[i - 1] != nums[i]:
                nums[result] = nums[i]
                result += 1
                
        return result

# %%
def removeElement( A, elem):
        # write your code here
        for i in range(len(A)):
            if elem in A:
                
                A.remove(elem)

        return len(A)

# %%
removeElement([0,4,4,0,0,2,4,4],4)

# %%
exclude = set(string.punctuation)
s='ab'
s = ''.join(ch for ch in s if ch not in exclude)
s=s.lower()
s=s.split()
s=''.join(s)

# %%
print(s[::-1])

# %%

'''del all punctuation'''
import string
class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """
    def isPalindrome(self, s):
        # write your code here
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        s=s.lower()
        s=s.split()
        s=''.join(s)
        
        if s==s[::-1]:
            return True
        else: 
            return False
#%%            

def deduplication( nums):
        # Write your code here
        n = len(nums)
        if n == 0:
            return 0
            
        nums.sort()
        result = 1
        for i in range(1, n):
            if nums[i - 1] != nums[i]:
                nums[result] = nums[i]
                result += 1
                
        return result


#%%
def removeElement( A, elem):
        n = len(A)
        l, r = 0, n - 1 
        while l <= r:
            if A[l] == elem:
                A[l], A[r] =A[r], A[l]
                r -= 1
            else:
                l += 1
        return A,r + 1

# %%
#敏锐洞察到这个狗比双指针可能是个好东西所以本大哥决定给丫记住草泥马的

def removeElement( A, elem):

    l,r=0,len(A)-1
    for i in range(len(A)):
        if A[l]==elem:
            A[l],A[r]=A[r],A[l]
            r-=1
        else:
            l+=1
    return A


# %%

class Solution:
    # @param head: the head of linked list.
    # @param val: an integer
    # @return: a linked node or null
    def findNode(self, head, val):
        # Write your code here
        while head is not None:
            if head.val == val:
                return head
            head = head.next
        return None



#%%


class Stack:
    """
    @param: x: An integer
    @return: nothing
    """
    
    def __init__(self,stack):
        self.stack = stack
        
    def push(self, x):
        # write your code here
        self.stack.append(x)

    """
    @return: nothing
    """
    def pop(self):
        # write your code here
        self.stack.pop()

    """
    @return: An integer
    """
    def top(self):
        # write your code here
        return self.stack[-1]

    """
    @return: True if the stack is empty
    """
    def isEmpty(self):
        # write your code here
        return len(self.stack) == 0


# %%
from queue import Queue
que=Queue()

# %%

for i in range(51):

    que.put(i)



# %%
que.get()

# %%
que.qsize()

# %%
que.full()

# %%
que.empty()

# %%
class ListNode:
    def __init__(self,val):
        self.val=val
        self.next=None

class MyLinkedList:
    def __init__(self, head):
        self.head=None
    
    def get(self, location):
        cur=self.head
        for i in range(location):
            cur=cur.next
        return cur.val
    
    def add(self,location,val):
        cif location>0:
        pre=self.head
        for i in range(location-1):
            pre=pre.next
        new_node=ListNode(val)
        new_node.next=pre.next
        pre.next=new_node

        elif location==0:
            new_node=ListNode(val)
            new_node.next=self.head
            self.head=new_node

# %%
import python_email

# %%
import os

# %%
import pandas as pd

# %%
df['no_cumulative'] = df.groupby(['name'])['no'].apply(lambda x: x.cumsum())

# %%
ass=25.5*45
cur_val=900

loss=ass-cur_val

# %%
from instagram.client import InstagramAPI

access_token = "YOUR_ACCESS_TOKEN"
client_secret = "YOUR_CLIENT_SECRET"
api = InstagramAPI(access_token=access_token, client_secret=client_secret)
recent_media, next_ = api.user_recent_media(user_id="userid", count=10)
for media in recent_media:
   print media.caption.text


#%%


# %%
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
n = 1018
pnull = .52
phat = .56
sm.stats.proportions_ztest(phat * n, n, pnull, alternative='larger')

# %%
import numpy as np
p1=np.random.binomial(1,0.368 , 247)
p2=np.random.binomial(1,0.389 , 308)
sm.stats.ttest_ind(p1, p2)


# %%
from datetime import datetime
#The strftime() method returns a string representing date and 
# time using date, time or datetime object.

now = datetime.now() # current date and time

year = now.strftime("%Y")
print("year:", year)

month = now.strftime("%m")
print("month:", month)

day = now.strftime("%d")
print("day:", day)

time = now.strftime("%H:%M:%S")
print("time:", time)

date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
print("date and time:",date_time)	

# %%
