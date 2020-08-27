# Pandas Use Guide

- 统计每行中，特征取值为-1的比例

```python
df["missing_feat"]=np.sum((df[cols] == -1).values, axis=1)
```

- cols是特征名称，返回其中稀疏类别特征的索引
```python
cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]
```

