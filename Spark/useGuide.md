# UserGuide

### groupby分组后，对组内的数据按照某些字段排序并生成列表

应用场景：比如需要统计用户浏览了那些文章，并将这些文章根据浏览时间排序形成对应的列表

```scala
scala> temp.show()
+--------+-----------+----+
|ClientID|    Product|Date|
+--------+-----------+----+
|     100|    Shampoo|   2|
|     101|       Book|   6|
|     100|Conditioner|   9|
|     101|   Bookmark|   3|
|     100|      Cream|   1|
|     101|      Book2|   4|
+--------+-----------+----+


scala> val temp1=temp.groupBy("ClientID").agg(sort_array(collect_list(struct("Date","Product"))) as "list")
temp1: org.apache.spark.sql.DataFrame = [ClientID: int, list: array<struct<Date:int,Product:string>>]

scala> temp1.show()
+--------+--------------------+
|ClientID|                list|
+--------+--------------------+
|     101|[[3, Bookmark], [...|
|     100|[[1, Cream], [2, ...|
+--------+--------------------+


scala> val temp1=temp.groupBy("ClientID").agg(sort_array(collect_list(struct("Date","Product"))) as "list").withColumn("date_list",concat_ws("|",col("list.Date"))).withColumn("product_list",concat_ws("|",col("list.Product")))
temp1: org.apache.spark.sql.DataFrame = [ClientID: int, list: array<struct<Date:int,Product:string>> ... 2 more fields]

scala> temp1.show()
+--------+--------------------+---------+--------------------+
|ClientID|                list|date_list|        product_list|
+--------+--------------------+---------+--------------------+
|     101|[[3, Bookmark], [...|    3|4|6| Bookmark|Book2|Book|
|     100|[[1, Cream], [2, ...|    1|2|9|Cream|Shampoo|Con...|
+--------+--------------------+---------+--------------------+

```