# Pandas Use Guide

- 创建dataframe

```python
index=['jin','qi','kui','xiang','lu','peng']
num_arr = np.random.randn(6,4) # 传入 numpy 随机数组
columns = ['A','B','C','D'] # 将列表作为列名
df = pd.DataFrame(num_arr, index = index, columns = columns)


data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(data, index=labels)
```

- 统计每行中，特征取值为-1的比例

```python
df["missing_feat"]=np.sum((df[cols] == -1).values, axis=1)
```

- cols是特征名称，返回其中稀疏类别特征的索引

```python
cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]
```

- df相关操作
    
    - loc与iloc的区别
    
    主要是loc只能根据行标签名字来索引相关行，iloc只能用行位置编号（int型数值）索引数据
    
    - loc与iloc可以进行行、列索引，也可以进行行列切片
    
    行切片： df.loc['A':'G', :]
    
    列切片： df.loc[:, ['animal', 'age']]
    
- Series能用apply操作，如果要对dataframe df进行apply操作，可以通过df.apply(lambda row,function(row),aixs=1)


```python
import pandas as pd

#展示df的前3行
df.iloc[:3]
# 方法二
#df.head(3)

#取出索引为[3, 4, 8]行的animal和age列
df.loc[df.index[[3, 4, 8]], ['animal', 'age']]

#取出df的animal和age列
df.loc[:, ['animal', 'age']]
# 方法二
# df[['animal', 'age']]

#取出age值大于3的行
df[df['age'] > 3]

#取出age值缺失的行
df[df['age'].isnull()]

#取出age在2,4间的行（不含）
df[(df['age']>2) & (df['age']<4)]
# 方法二
# df[df['age'].between(2, 4)]

#f行的age改为1.5
df.loc['f', 'age'] = 1.5

#计算每个不同种类animal的age的平均数
df.groupby('animal')['age'].mean()

#计算df中每个种类animal的数量
df['animal'].value_counts()

#先按age降序排列，后按visits升序排列
df.sort_values(by=['age', 'visits'], ascending=[False, True])

#将priority列中的yes, no替换为布尔值True, False
df['priority'] = df['priority'].map({'yes': True, 'no': False})

#统计每列中null的个数
NAs=pd.concat([train.isnull().sum(),test.isnull().sum()],axis=1,keys=['train','test'])
NAs[NAs.sum(axis=1)>0]
```

### df进阶操作

unstack操作与pivot操作：unstack()方法是针对索引或者标签的，即将列索引转成最内层的行索引；而pivot()方法则是针对列的值，即指定某列的值作为行索引，指定某列的值作为列索引，然后再指定哪些列作为索引对应的值。unstack()针对索引进行操作，pivot()针对值进行操作。更直观的，unstack是将某一列索引变为行索引，stack是将行索引变为列索引。

```python
data=pd.DataFrame(np.arange(6).reshape((2,3)),index=pd.Index(['street1','street2']),columns=pd.Index(['one','two','three']))
data2=data.stack()
#street1  one      0
#         two      1
#         three    2
#street2  one      3
#         two      4
data3=data2.unstack()

#         one  two  three
#street1    0    1      2
#street2    3    4      5
```



- 一个全数值DatraFrame，每个数字减去该行的平均数
    
```python
df = pd.DataFrame(np.random.random(size=(5, 3)))
print(df)
df1 = df.sub(df.mean(axis=1), axis=0)
print(df1)
```

- 给定DataFrame，求A列每个值的前3大的B的值的和

```python
df = pd.DataFrame({'A': list('aaabbcaabcccbbc'), 
                   'B': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})
print(df)
df1 = df.groupby('A')['B'].nlargest(3).sum(level=0)
print(df1)
```

- 给定DataFrame，有列A, B，A的值在1-100（含），对A列每10步长，求对应的B的和
```python
df = pd.DataFrame({'A': [1,2,11,11,33,34,35,40,79,99], 
                   'B': [1,2,11,11,33,34,35,40,79,99]})
print(df)
df1 = df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()
print(df1)
```

- 一个全数值的DataFrame，返回最大3个值的坐标

注意层级化dataframe的使用

```python
df = pd.DataFrame(np.random.random(size=(5, 3)))
print(df)
df.unstack().sort_values()[-3:].index.tolist()
```

- 给定DataFrame，将负值代替为同组的平均值

```python
df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [-12,345,3,1,45,14,4,-52,54,23,-235,21,57,3,87]})
print(df)

def replace(group):
    mask = group<0
    group[mask] = group[~mask].mean()
    return group

df['vals'] = df.groupby(['grps'])['vals'].transform(replace)
print(df)
```

- 计算3位滑动窗口的平均值，忽略NAN

```python
df = pd.DataFrame({'group': list('aabbabbbabab'),
                    'value': [1, 2, 3, np.nan, 2, 3, np.nan, 1, 7, 3, np.nan, 8]})
print(df)

g1 = df.groupby(['group'])['value']
g2 = df.fillna(0).groupby(['group'])['value'] 

s = g2.rolling(3, min_periods=1).sum() / g1.rolling(3, min_periods=1).count()
s.reset_index(level=0, drop=True).sort_index()
```

- 数据预处理

```python
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})

#缺失值处理
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)

#将From_To列从_分开，分成From, To两列，并删除原始列
temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']
df = df.join(temp)
df = df.drop('From_To', axis=1)

#Airline列，有一些多余的标点符号，需要提取出正确的航司名称。举例：'(British Airways. )' 应该改为 'British Airways'.
df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip()

#Airline列，数据被以列表的形式录入，但是我们希望每个数字被录入成单独一列，delay_1, delay_2, ...没有的用NAN替代。
delays = df['RecentDelays'].apply(pd.Series)
delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns)+1)]
df = df.drop('RecentDelays', axis=1).join(delays)
```

### 可视化

- 对目标变量的分布，查看label分布是否不均衡

- 对numeric数据，boxplot箱线图绘制。查看是否偏态----如果偏态，需要转为正太

- 对于坐标类数据，用scatter plot来查看分布

- 对于分类问题，将数据根据不同类别着色

- 绘制相关系数表


