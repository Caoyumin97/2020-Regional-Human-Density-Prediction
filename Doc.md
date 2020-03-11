# Documentation

# 1 数据文件分析

## 1.1 area_passenger_info.csv

区域属性描述文件

![image-20200305221614840](C:\Users\Cao\AppData\Roaming\Typora\typora-user-images\image-20200305221614840.png)

## 1.2 area_passenger_index.csv

区域历史人流量指数

![image-20200305221857827](C:\Users\Cao\AppData\Roaming\Typora\typora-user-images\image-20200305221857827.png)

## 1.3 grid_strength.csv

网格间联系强度

![image-20200305222027354](C:\Users\Cao\AppData\Roaming\Typora\typora-user-images\image-20200305222027354.png)

## 1.4 migration_index.csv

省市级迁徙指数

![image-20200305222143152](C:\Users\Cao\AppData\Roaming\Typora\typora-user-images\image-20200305222143152.png)

## 1.5 shortstay_date.csv

网格人流量指数

![image-20200305222239646](C:\Users\Cao\AppData\Roaming\Typora\typora-user-images\image-20200305222239646.png)



# 2 赛题分析

## 2.1 规则

根据以上历史数据（2020.01.17-2020.02.15），预测各区域未来9天（2020.02.16-2020.02.24）各区域（area-level）时变的（hour-dependent）人流指数。

* 区域：占地面积较大，最大达21 $\text{km}^2$

* 网格：按经纬度划分的矩形网格，面积为$\text{200m}\times\text{200m}=0.04\text{ km}^2$

## 2.2 评价指标

$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\\
0\le score = \frac{1}{1+RMSE}\le1
$$

## 2.3 初步思路：

1. 时间序列特征

2. 建立“区域画像”：

   * area embedding (100-dim)

   * location: distance to CBD/traffic center/other area/etc. (30-dim)

   * time-of-day, day-of-week, days-from-hot-day [decaying positional encoding ?] (5~8-dim)

     *hot days: recognize from migration index, grid activity strength, etc.

   * raw time-series (6~8-dim)

   * grid-based features (~20-dim)

