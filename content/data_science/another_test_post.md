Title: It Recently Occurred to Me...
Date: 2018-09-09
Tags: musings

... I should get back into writing/blogging things that come to mind. For example, that type annotations and type-checking with `mypy` are great tools for data scientists as well as day-to-day developers - e.g.,

```python
def circle_area(radius: float) -> float:
    pi = 3.141
    return pi * (r ** 2)
```

Makes it so much easier to understand how to use functions and methods as well as providing extra piece of mind that code will work as expected.