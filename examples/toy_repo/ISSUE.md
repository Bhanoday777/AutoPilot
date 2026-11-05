# `median()` returns the wrong value for even-length lists

**Describe the bug**
`calc.stats.median` gives the wrong answer when the list has an even number of
elements. It seems to return the upper of the two middle values instead of
their average.

**To reproduce**
```python
>>> from calc.stats import median
>>> median([1, 2, 3, 4])
3
```

**Expected behaviour**
```python
>>> median([1, 2, 3, 4])
2.5
```
The median of an even-length list should be the mean of the two middle values.

**Environment**
- Python 3.11
- calc @ main
