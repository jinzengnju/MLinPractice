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