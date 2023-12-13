# Lab6 submission
Put your writeup and codes here



## Module 1

```c
void setup()
{
    for (int i = 0; i < 256; i++)
    {
        clflush(UserArray + (i * space));
    }
}

int main()
{
    UserArray = malloc(256 * space);
    memset(UserArray, 0, 256 * space);

    uint32_t failed = 0;
    for (int i = 0; i < totalNum; i++)
    {
        uint8_t data = random() & 0xff;
        setup();
        maccess(UserArray + (data * space));
        for (int j = 0; j < 256; j++)
        {
            uint32_t timing = reload(UserArray + (j * space));
            if (timing < threshold && j != data)
            {
                printf("i %d, j %d, t %u\n", i, j, timing);
                failed++;
                break;
            }
        }
    }
    printf("failed: %u/%u\n", failed, totalNum);
}
```

The `threshold` is setted to `150` on the workstation, and it shows all data have been received correctly.

![](failedNum.png)




## Module 2

![](segfault.png)

With option `-s`, the program setup a `SIGSEGV` handler, and print a message `segfault suppressed` after Segmentation fault occurred. Without option `-s`, a Segmentation fault will killed the program.


## Module 3

The CPU model is shown as following
```sh
cat /proc/cpuinfo | grep "model name"

model name      : Intel(R) Core(TM) i5-4590 CPU @ 3.30GHz
```

## Module 4

the kernel message is

```
Have a great winter break! You deserve it after you see this message
```