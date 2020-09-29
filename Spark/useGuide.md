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

###分组后组内按照某个字段排序

```scala
import java.sql.Date
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object SortQuestion extends App{

  val spark = SparkSession.builder().appName("local").master("local[*]").getOrCreate()
  import spark.implicits._
  case class ABC(a: Int, b: Int, c: Int)

  val first = Seq(
    ABC(1, 2, 3),
    ABC(1, 3, 4),
    ABC(2, 4, 5),
    ABC(2, 5, 6)
  ).toDF("a", "b", "c")

  val second = Seq(
    (1, 2, (Date.valueOf("2018-01-02"), 30)),
    (1, 3, (Date.valueOf("2018-01-01"), 20)),
    (2, 4, (Date.valueOf("2018-01-02"), 50)),
    (2, 5, (Date.valueOf("2018-01-01"), 60))
  ).toDF("a", "b", "c")

  first.join(second.withColumnRenamed("c", "c2"), Seq("a", "b")).groupBy("a").agg(sort_array(collect_list("c2")))
    .show(false)

}
```

按照(java.sql.Date, Int)中的Int进行排序

```
case class Result(a: Int, b: Int, c: Int, c2: (java.sql.Date, Int))

val joined = first.join(second.withColumnRenamed("c", "c2"), Seq("a", "b"))
joined.as[Result]
  .groupByKey(_.a)
  .mapGroups((key, xs) => (key, xs.map(_.c2).toSeq.sortBy(_._2)))
  .show(false)

// +---+----------------------------------+            
// |_1 |_2                                |
// +---+----------------------------------+
// |1  |[[2018-01-01,20], [2018-01-02,30]]|
// |2  |[[2018-01-02,50], [2018-01-01,60]]|
// +---+----------------------------------+
```

- 这里的xs.map(_.c2)相当于取出c2的元素

- joined.as[Result] 是将DataFrame转换为DataSet

```
val sortUdf = udf { (xs: Seq[Row]) => xs.sortBy(_.getAs[Int](1) )
                                        .map{ case Row(x:java.sql.Date, y: Int) => (x,y) }}

first.join(second.withColumnRenamed("c", "c2"), Seq("a", "b"))
     .groupBy("a")
     .agg(sortUdf(collect_list("c2")))
     .show(false)

//+---+----------------------------------+
//|a  |UDF(collect_list(c2, 0, 0))       |
//+---+----------------------------------+
//|1  |[[2018-01-01,20], [2018-01-02,30]]|
//|2  |[[2018-01-02,50], [2018-01-01,60]]|
//+---+----------------------------------+
```

- 这里使用了.map{ case Row(x:java.sql.Date, y: Int) => (x,y)，如果不使用，返回的仍然是Seq\[Row\]类型，这里使用map是返回(x,y)的元组

### 将多个数组分为多行，元素索引相互对应

```python
from pyspark import Row
from pyspark.sql import SQLContext
from pyspark.sql.functions import explode

sqlc = SQLContext(sc)

df = sqlc.createDataFrame([Row(a=1, b=[1,2,3],c=[7,8,9], d='foo')])
# +---+---------+---------+---+
# |  a|        b|        c|  d|
# +---+---------+---------+---+
# |  1|[1, 2, 3]|[7, 8, 9]|foo|
# +---+---------+---------+---+

#想得到一下结果

+---+---+----+------+
|  a|  b|  c |    d |
+---+---+----+------+
|  1|  1|  7 |  foo |
|  1|  2|  8 |  foo |
|  1|  3|  9 |  foo |
+---+---+----+------+

```
- 方法一

```python
from pyspark.sql.functions import arrays_zip, col, explode

(df
    .withColumn("tmp", arrays_zip("b", "c"))
    .withColumn("tmp", explode("tmp"))
    .select("a", col("tmp.b"), col("tmp.c"), "d"))
```

- 方法二：

```python
(df
    .rdd
    .flatMap(lambda row: [(row.a, b, c, row.d) for b, c in zip(row.b, row.c)])
    .toDF(["a", "b", "c", "d"]))
```

- 方法三：

```python
from pyspark.sql.functions import array, struct

# SQL level zip of arrays of known size
# followed by explode
tmp = explode(array(*[
    struct(col("b").getItem(i).alias("b"), col("c").getItem(i).alias("c"))
    for i in range(n)
]))

(df
    .withColumn("tmp", tmp)
    .select("a", col("tmp").getItem("b"), col("tmp").getItem("c"), "d"))
```

- 方法四

```python
def zip_and_explode(*colnames, n):
    return explode(array(*[
        struct(*[col(c).getItem(i).alias(c) for c in colnames])
        for i in range(n)
    ]))

df.withColumn("tmp", zip_and_explode("b", "c", n=3))
```

### zip多个数组

将多个数组对应元素按照索引zip在一起

```
val ints = List(1,2,3)
val chars = List('a', 'b', 'c')
val strings = List("Alpha", "Beta", "Gamma")
val bools = List(true, false, false)

ints zip chars zip strings zip bools
List(ints, chars, strings, bools).transpose

val ints = List(1,2,3)
val chars = List('a', 'b', 'c')
val strings = List("Alpha", "Beta", "Gamma")
val bools = List(true, false, false)
(ints zip chars zip strings zip bools) map { case (((i,c),s),b) => (i,c,s,b)}

```

