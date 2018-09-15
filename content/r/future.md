Title: Asynchronous and Distributed Programming in R with the Future Package
Date: 2016-11-02
Tags: data-processing, high-performance-computing

![Future!][future_pic]

Every now and again someone comes along and writes an R package that I consider to be a 'game changer' for the language and it's application to Data Science. For example, I consider [dplyr][dplyr] one such package as it has made data munging/manipulation _that_ more intuitive and more productive than it had been before. Although I only first read about it at the beginning of this week, my instinct tells me that in [Henrik Bengtsson's][HenrikBengtsson] [future][future] package we might have another such game-changing R package.

The [future][future] package provides an API for futures (or promises) in R. To quote Wikipedia, a [future or promise][future_and_promises] is,

> _... a proxy for a result that is initially unknown, usually because the computation of its value is yet incomplete._

A classic example would be a request made to a web server via HTTP, that has yet to return and whose value remains unknown until it does (and which has promised to return at some point in the future). This 'promise' is an object assigned to a variable in R like any other, and allows code execution to progress until the moment the code explicitly requires the future to be resolved (i.e. to 'make good' on it's promise). So the code does not need to wait for the web server until the very moment that the information anticipated in its response it actually needed. In the intervening execution time we can send requests to other web servers, run some other computations, etc. Ultimately, this leads to **faster** and **more efficient code**. This way of working also opens the door to distributed (i.e. parallel) computation, as the computation assigned to each new future can be executed on a new thread (and executed on a different core on the same machine, or on another machine/node).

The future API is extremely expressive and the associated documentation is excellent. My motivation here is not to repeat any of this, but rather to give a few examples to serve as inspiration for how futures could be used for day-to-day Data Science tasks in R.

# Creating a Future to be Executed on a Different Core to that Running the Main Script
To demonstrate the syntax and structure required to achieve this aim, I am going to delegate to a future the task of estimating the mean of 10 million random samples from the normal distribution, and ask it to spawn a new R process on a different core in order to do so. The code to achieve this is as follows,
```r
library(future)

f <- future({
  samples <- rnorm(10000000)
  mean(samples)
}) %plan% multiprocess
w <- value(f)
w
# [1] 3.046653e-05
```
- `future({...})` assigns the code (actually a construct known as a [closure][closure]), to be computed asynchronously from the main script. The code will be start execution the moment this initial assignment is made;
- `%plan% multiprocess` sets the future's execution plan to be on a different core (or thread); and,
- `value` asks for the return value of future. This will block further code execution until the future can be resolved.

The above example can easily be turned into a function that outputs dots (`...`) to the console until the future can be resolved and return it's value,
```r
f_dots <- function() {
  f <- future({
    s <- rnorm(10000000)
    mean(s)
  }) %plan% multiprocess

  while (!resolved(f)) {
    cat("...")
  }
  cat("\n")

  value(f)
}
f_dots()
# ............
# [1] -0.0001872372
```
Here, `resolved(f)` will return `FALSE` until the future `f` has finished executing.

# Useful Use Cases
I can recall many situations where futures would have been handy when writing R scripts. The examples below are the most obvious that come to mind. No doubt there will be many more.

## Distributed (Parallel) Computation
In the past, when I've felt the need to distribute a calculation I have usually used the `mclapply` function (i.e. multi-core `lapply`), from the `parallel` library that comes bundled together with base R. Computing the mean of 100 million random samples from the normal distribution would look something like,
```r
library(parallel)

sub_means <- mclapply(
              X = 1:4,
              FUN = function(x) { samples <- rnorm(25000000); mean(samples) },
              mc.cores = 4)

final_mean <- mean(unlist(sub_mean))
final_mean
# [1] -0.0002100956
```
Perhaps more importantly, the script will be 'blocked' until `sub_means` has finished executing. We can achieve the same end-result, but without blocking, using futures,
```r
single_thread_mean <- function() {
  samples <- rnorm(25000000)
  mean(samples)
}

multi_thread_mean <- function() {
  f1 <- future({ single_thread_mean() }) %plan% multiprocess
  f2 <- future({ single_thread_mean() }) %plan% multiprocess
  f3 <- future({ single_thread_mean() }) %plan% multiprocess
  f4 <- future({ single_thread_mean() }) %plan% multiprocess

  mean(value(f1), value(f2), value(f3), value(f4))
}

multi_thread_mean()
# [1] -4.581293e-05
```
We can compare computation time between the single and multi-threaded versions of the mean computation (using the [microbenchmark][microbenchmark] package),
```r
library(microbenchmark)

microbenchmark({ samples <- rnorm(100000000); mean(samples) },
               multi_thread_mean(),
               times = 10)
# Unit: seconds
#                  expr      min       lq     mean   median       uq      max neval
#  single_thread(1e+08) 7.671721 7.729608 7.886563 7.765452 7.957930 8.406778    10
#   multi_thread(1e+08) 2.046663 2.069641 2.139476 2.111769 2.206319 2.344448    10
```
We can see that the multi-threaded version is nearly 3 times faster, which is not surprising given that we're using 3 extra threads. Note that time is lost spawning the extra threads and combining their results (usually referred to as 'overhead'), such that distributing a calculation can actually increase computation time if the benefit of parallelisation is less than the cost of the overhead.

## Non-Blocking Asynchronous Input/Output
I have often found myself in the situation where I need to read several large CSV files, each of which can take a long time to load. Because the files can only be loaded sequentially, I have had to wait for one file to be read before the next one can start loading, which compounds the time devoted to input. Thanks to futures, we can can now achieve [asynchronous input and output][asyncio] as follows,
```r
library(readr)

df1 <- future({ read_csv("data/csv1.csv") }) %plan% multiprocess
df2 <- future({ read_csv("data/csv2.csv") }) %plan% multiprocess
df3 <- future({ read_csv("data/csv3.csv") }) %plan% multiprocess
df4 <- future({ read_csv("data/csv4.csv") }) %plan% multiprocess

df <- rbind(value(df1), value(df2), value(df3), value(df4))
```
Running `microbenchmark` on the above code illustrates the speed-up (each file is ~50MB in size),
```r
# Unit: seconds
#                   min       lq     mean   median       uq      max neval
#  synchronous 7.880043 8.220015 8.502294 8.446078 8.604284 9.447176    10
# asynchronous 4.203271 4.256449 4.494366 4.388478 4.490442 5.748833    10
```

The same pattern can be applied to making HTTP requests asynchronously. In the following example I make an asynchronous HTTP GET request to the OpenCPU public API, to retrieve the Boston housing dataset via JSON. While I'm waiting for the future to resolve the response I keep making more asynchronous requests, but this time to `http://time.jsontest.com` to get the current time. Once the original future has resolved, I block output until all remaining futures have been resolved.
```r
library(httr)
library(jsonlite)

time_futures <- list()

data_future <- future({
  response <- GET("http://public.opencpu.org/ocpu/library/MASS/data/Boston/json")
  fromJSON(content(response, as = "text"))
}) %plan% multiprocess

while (!resolved(data_future)) {
  time_futures <- append(time_futures, future({ GET("http://time.jsontest.com") }) %plan% multiprocess)
}
values(time_futures)
# [[1]]
# Response [http://time.jsontest.com/]
#   Date: 2016-11-02 01:31
#   Status: 200
#   Content-Type: application/json; charset=ISO-8859-1
#   Size: 100 B
# {
#    "time": "01:31:19 AM",
#    "milliseconds_since_epoch": 1478050279145,
#    "date": "11-02-2016"
# }

head(value(data_future))
# crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat medv
# 1 0.0063 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98 24.0
# 2 0.0273  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14 21.6
# 3 0.0273  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03 34.7
# 4 0.0324  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94 33.4
# 5 0.0690  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33 36.2
# 6 0.0298  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21 28.7
```

The same logic applies to accessing databases and executing SQL queries via [ODBC][odbc] or [JDBC][jdbc]. For example, large complex queries can be split into 'chunks' and sent asynchronously to the database server in order to have them executed on multiple server threads. The output can then be unified once the server has sent back the chunks, using R (e.g. with [dplyr][dplyr]). This is a strategy that I have been using with Apache Spark, but I could now implement it within R. Similarly, multiple database tables can be accessed concurrently, and so on.  

# Final Thoughts
I have only really scratched the surface of what is possible with futures. For example, [future][future] supports multiple execution plans including `lazy` and `cluster` (for multiple machines/nodes) - I have only focused on increasing performance on a single machine with multiple cores. If this post has provided some inspiration or left you curious, then head over to the official [future docs][future] for the full details (which are a joy to read and work-through).


[future_pic]: {filename}/images/r/future/the_future.jpg "the_future"

[dplyr]: https://github.com/hadley/dplyr "dplyr on GitHub"

[HenrikBengtsson]: https://www.linkedin.com/in/henrikbengtsson "Henrik Bengtsson on LinkedIn"

[future]: https://github.com/HenrikBengtsson/future "future package in GitHub"

[future_and_promises]: https://en.wikipedia.org/wiki/Futures_and_promises "Wikipedia on futures and promises"

[closure]: http://adv-r.had.co.nz/Functional-programming.html#closures "Hadley Wickham on closures"

[microbenchmark]: https://cran.r-project.org/web/packages/microbenchmark/index.html "microbenchmark on CRAN"

[asyncio]: https://en.wikipedia.org/wiki/Asynchronous_I/O "Wikipedia on asynchronous io"

[odbc]: https://en.wikipedia.org/wiki/Open_Database_Connectivity "Wikipedia on ODBC"

[jdbc]: https://en.wikipedia.org/wiki/Java_Database_Connectivity "Wikipedia on JDBC"
