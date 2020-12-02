# C语言程序设计

### C语言数据类型

- 基本类型：整数类型与浮点类型

- 枚举类型：定义在程序中只能赋予其一定的离散值的变量

- void类型

    - 函数返回为空
    
    - 指针指向void：类型为void * 的指针代表对象的地址，而不是类型。例如，内存分配函数void *malloc(size_t size)；返回指向void的指针，可以转为任何数据类型

- 派生类型：指针类型、数组类型、结构类型、共用体类型和函数类型

### C语言变量

变量是程序可操作存储区的名称，每个变量都有特定的类型，类型决定了变量存储的大小和布局。变量声明有以下两种方法：

- 需要建立存储空间

例如：int a；在声明的时候已经建立了存储空间

- 不需建立存储空间

使用extern关键字声明变量而不是定义它。比如，extern int a，其中变量a可以在别的文件中定义.如果要在一个源文件引入另外一个源文件中定义的变量，加入extern关键字即可

```
addtwonum.c 文件代码:
#include <stdio.h>
/*外部变量声明*/
extern int x ;
extern int y ;
int addtwonum()
{
    return x+y;
}

test.c 文件代码：
#include <stdio.h>
/*定义两个全局变量*/
int x=1;
int y=2;
int addtwonum();
int main(void)
{
    int result;
    result = addtwonum();
    printf("result 为: %d\n",result);
    return 0;
}
```

**需要注意的是，除非有extern关键字，否则都是关于变量的定义**

```
extern int i; //声明，不是定义
int i; //声明，也是定义
```

### C常量

是固定值，在程序执行期间不会改变。这些固定的值，又叫做字面量。定义常量有以下两种方法：

- \#define预处理器

- const关键字

```
#include <stdio.h>
 
#define LENGTH 10   
#define WIDTH  5
#define NEWLINE '\n'

#include <stdio.h>
 
int main()
{
   const int  LENGTH = 10;
   const int  WIDTH  = 5;
   const char NEWLINE = '\n';
   int area;  
   area = LENGTH * WIDTH;
   printf("value of area : %d", area);
   printf("%c", NEWLINE);
   return 0;
}
```

### C存储类

- auto

auto 只能用在函数内，即 auto 只能修饰局部变量。

- register

定义存储在寄存器而不是RAM中的局部变量，变量的最大尺寸等于寄存器的大小。**寄存器只用于需要快速访问的变量，比如计数器**。需要注意的是，定义“register”并不意味着变量将被存储在寄存器中，它意味着变量可能存储在寄存器中。

- static

使用static修饰的局部变量可以在函数调用之间保持**局部变量的值**。static修饰符也可以用于全局变量，当static修饰全局变量时，会使得变量的作用域限制在它声明的文件内。

全局声明的一个 static 变量或方法可以被任何函数或方法调用，只要这些方法出现在跟 static 变量或方法同一个文件中。实例中 count 作为全局变量可以在函数内使用，thingy 使用 static 修饰后，不会在每次调用时重置。

```
#include <stdio.h>
 
/* 函数声明 */
void func1(void);
 
static int count=10;        /* 全局变量 - static 是默认的 */
 
int main()
{
  while (count--) {
      func1();
  }
  return 0;
}
 
void func1(void)
{
/* 'thingy' 是 'func1' 的局部变量 - 只初始化一次
 * 每次调用函数 'func1' 'thingy' 值不会被重置。
 */                
  static int thingy=5;
  thingy++;
  printf(" thingy 为 %d ， count 为 %d\n", thingy, count);
}
```

- extern

```
第一个文件：main.c
#include <stdio.h>
 
int count ;
extern void write_extern();
 
int main()
{
   count = 5;
   write_extern();
}

第二个文件：support.c
#include <stdio.h>
 
extern int count;
 
void write_extern(void)
{
   printf("count is %d\n", count);
}
```

### C函数

函数定义与函数声明：函数定义需要写出完整的函数逻辑，函数声明即申明应该如何调用函数即可，即返回类型、函数参数类型。当您在一个源文件中定义函数且在另一个文件中调用函数时，函数声明是必须的。**这种情况下，应该在调用函数的文件顶部声明函数**

```
/* 函数定义，函数返回两个数中较大的那个数 */
int max(int num1, int num2) 
{
   /* 局部变量声明 */
   int result;
   if (num1 > num2)
      result = num1;
   else
      result = num2;
   return result; 
}

int max(int num1, int num2);
int max(int, int);

```

```
#include <stdio.h>
 
/* 函数声明 */
int max(int num1, int num2);
 
int main ()
{
   /* 局部变量定义 */
   int a = 100;
   int b = 200;
   int ret;
 
   /* 调用函数来获取最大值 */
   ret = max(a, b);
 
   printf( "Max value is : %d\n", ret );
 
   return 0;
}
 
/* 函数返回两个数中较大的那个数 */
int max(int num1, int num2) 
{
   /* 局部变量声明 */
   int result;
 
   if (num1 > num2)
      result = num1;
   else
      result = num2;
 
   return result; 
}
```

- 函数参数：形式参数，是函数内部的局部变量，在进入函数时被创建，退出函数时被销毁。而在调用函数时，有两种向函数传递参数的方式：

    - 传值调用：把参数的实际值复制给函数的形式参数。在这种情况下，修改函数内的形式参数不会影响实际参数。
    
    - 引用调用：通过指针传递方式，形参为指向实参地址的指针，当对形参的指向操作时，就相当于对实参本身进行的操作。这种方法有什么好处呢？**传递指针可以让多个函数访问指针所引用的对象，而不用把对象声明为全局可访问**
```
/* 函数定义 */
void swap(int *x, int *y)
{
   int temp;
   temp = *x;    /* 保存地址 x 的值 */
   *x = *y;      /* 把 y 赋值给 x */
   *y = temp;    /* 把 temp 赋值给 y */
  
   return;
}

#include <stdio.h>
 
/* 函数声明 */
void swap(int *x, int *y);
 
int main ()
{
   /* 局部变量定义 */
   int a = 100;
   int b = 200;
   printf("交换前，a 的值： %d\n", a );
   printf("交换前，b 的值： %d\n", b );
   /* 调用函数来交换值
    * &a 表示指向 a 的指针，即变量 a 的地址
    * &b 表示指向 b 的指针，即变量 b 的地址
   */
   swap(&a, &b);
   printf("交换后，a 的值： %d\n", a );
   printf("交换后，b 的值： %d\n", b );
   return 0;
}

```

### C作用域规则

C语言中有三个地方可以声明变量：

- 局部变量

- 全局变量

- 形式参数

全局变量保存在内存的全局存储单元中，占用静态的存储单元；局部变量保存在栈中，**只有在所在函数被调用时才动态的为变量分配存储单元**

### 指针

![](/home/jin/git/MlinPractice/C++/pictures/1.png)

* 定义指针变量：int  *p;
* 把变量地址赋值给指针：p = &var_runoob
* 访问指针变量中可用地址的值：指针指向的内存单元中的值为，*p=10

#### NULL指针

NULL 指针是一个定义在标准库中的值为零的常量

~~~C
#include <stdio.h>
 
int main ()
{
   int  *ptr = NULL;
 
   printf("ptr 的地址是 %p\n", ptr  );
 
   return 0;
}
// ptr 的地址是 0x0
~~~

#### 指向指针的指针

指针的指针声明：int **var;

![](/home/jin/git/MlinPractice/C++/pictures/2.png)

~~~C
#include <stdio.h>
 
int main ()
{
   int  V;
   int  *Pt1;
   int  **Pt2;
 
   V = 100;
 
   /* 获取 V 的地址 */
   Pt1 = &V;
 
   /* 使用运算符 & 获取 Pt1 的地址 */
   Pt2 = &Pt1;
 
   /* 使用 pptr 获取值 */
   printf("var = %d\n", V );
   printf("Pt1 = %p\n", Pt1 );
   printf("*Pt1 = %d\n", *Pt1 );
    printf("Pt2 = %p\n", Pt2 );
   printf("**Pt2 = %d\n", **Pt2);
 
   return 0;
}

//var = 100
//Pt1 = 0x7ffee2d5e8d8
//*Pt1 = 100
//Pt2 = 0x7ffee2d5e8d0
//**Pt2 = 100
~~~

指针的指针，如果要取对应存储单元的值时，需要使用两重*运算符