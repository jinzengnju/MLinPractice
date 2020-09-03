# 特征工程

### 特征选择

- pandas画出相关性矩阵

```python
import pandas as pd
df = pd.DataFrame({'A': [1,2,11,11,33,34,35,40,79,99], 
                   'B': [1,2,11,11,33,34,35,40,79,99]})
corrmat=df.corr(method='spearman')
```