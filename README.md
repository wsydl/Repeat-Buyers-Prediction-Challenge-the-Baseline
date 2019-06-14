# Repeat Buyers Prediction Challenge the Baseline
## 大赛介绍：
阿里天池新人赛: Repeat Buyers Prediction Challenge the Baseline 

电商回头客预测 

## 赛题链接
> https://tianchi.aliyun.com/competition/entrance/231576/introduction?spm=5176.12281973.1005.1.3dd52448p8DbFE

## 赛题解读：
   * 数据：赛题提供商家和他们在“双十一”促销期间获得的相应的新买家数据；
      * 该数据集包含了匿名用户在“双十一”前后6个月的购物记录，以及表明他们是否重复购买的标签信息
   * 目标：预测这些新买家在6个月内再次从相同商家购买商品的概率；

## 赛题数据
   * 用户信息
   * 用户行为日志
   * 训练集
   * 测试集
## 赛题难点
赛题数据量非常大，提特征的过程中内存爆炸，如何处理这一过程是本次解决的过程中的难点
## constant.py
   * 列名常量
   * 路径常量 
   * 特征集 
   * 输出日志函数
   * month_action计数函数
   * 月份划分函数
## data_split.py
   * 扩展原始训练集和测试集
   * 划分训练集和验证集
##  get_feature
   * 用户特征 
   * 用户品牌特征
   * 用户-商户特征
   * 用户类别特征
   * 商户特征

   
