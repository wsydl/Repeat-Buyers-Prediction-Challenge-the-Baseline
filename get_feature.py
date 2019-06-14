from constant import *
from pathlib import Path
import pandas as pd
import numpy as np


def u_feat(user_log):
    logger = get_log("u_feat")
    # 用户点击/加入购物车/购买/加入到收藏计数特征
    logger.info("提取用户数据集点击/加入购物车/购买/加入到收藏计数特征...")

    sub = user_log[[COL_USER_ID, COL_ACTION_TYPE]]
    sub[COL_ACTION_TYPE] = sub[COL_ACTION_TYPE].astype('category')

    user_count_feature = pd.get_dummies(sub)
    user_action_count_feature = pd.pivot_table(user_count_feature, index=[COL_USER_ID],
                                               aggfunc='sum').reset_index().rename(
        columns={'action_type_0': 'user_click_count', 'action_type_1': 'user_add_to_cart_count',
                 'action_type_2': 'user_purchase_count', 'action_type_3': 'user_add_to_favorite_count'})

    # 计算总点击率/加入购物车率/购买率/加入到收藏计数特征率
    logger.info('提取用户点击率/加入购物车率/购买率/加入到收藏计数特征率...')

    # 首先计算总数
    user_action_count_feature['sum_count'] = user_action_count_feature.iloc[:, 1:].apply(sum, axis=1)

    # 列操作
    user_action_count_feature['user_click_rate'] = user_action_count_feature['user_click_count'] / \
                                                   user_action_count_feature['sum_count']
    user_action_count_feature['user_add_to_cart_rate'] = user_action_count_feature['user_add_to_cart_count'] / \
                                                         user_action_count_feature['sum_count']
    user_action_count_feature['user_purchase_rate'] = user_action_count_feature['user_purchase_count'] / \
                                                      user_action_count_feature['sum_count']
    user_action_count_feature['user_add_to_favorite_rate'] = user_action_count_feature['user_add_to_favorite_count'] / \
                                                             user_action_count_feature['sum_count']

    user_action_count_feature.to_csv(GEN_PATH + "feat/u/u_a_c_feat.csv", index=False)

    logger.info('提取用户每月点击/加入购物车/购买/加入到收藏计数特征...')
    part = user_log[[COL_USER_ID, COL_ACTION_TYPE, COL_TIME_STAMP]]

    # 2. 处理月份和操作类型
    logger.info("处理月份")
    part['month'] = part['time_stamp'].apply(month_classify)
    part = part.drop('time_stamp', axis=1)

    # 3.合并月份和操作类型 为分类做准备
    logger.info("合并月份和操作类型")
    part['act_month'] = pd.Series((repr(x) + "_" + y for x, y in zip(part['action_type'], part['month'])))
    part = part.drop(['action_type', 'month'], axis=1)

    # 4. one_hot编码
    logger.info("one_hot编码")
    dummies = pd.get_dummies(part[['user_id', 'act_month']])

    # 5. 数据透视 求每个月份区间每个用户各种操作数的均值、最大值、中位数、标准差和总数
    logger.info("数据透视")
    table = pd.pivot_table(dummies, index=['user_id'], aggfunc=[np.mean, np.max, np.median, np.std, np.sum])

    # 6. 把aggfunc带来的复合索引化为单层索引
    logger.info("把aggfunc带来的复合索引化为单层索引")
    table.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in table.columns]

    # 7. 输出特征
    logger.info("输出特征")
    table = table.reset_index()
    table.to_csv(GEN_PATH + "feat/u/u_a_s_feat.csv", index=False)


def u_b_feat(user_log):
    logger = get_log("u_b_feat")
    sub = user_log[[COL_USER_ID, COL_BRAND_ID, COL_ACTION_TYPE]]

    logger.info("提取用户对每个品牌的点击特征")
    sub[COL_ACTION_TYPE] = sub[COL_ACTION_TYPE].astype('category')
    u_b_f = pd.get_dummies(sub)

    u_b_a_c = pd.pivot_table(u_b_f, index=[COL_USER_ID, COL_BRAND_ID], aggfunc='sum').reset_index()

    u_b_a_c.to_csv(GEN_PATH + "feat/u_b/u_b_a_c_feat.csv", index=False)

    # 特征太大 注释
    # logger.info("提取用户每个月对每个品牌的点击特征")
    # part = user_log[[COL_USER_ID, COL_BRAND_ID, COL_ACTION_TYPE, COL_TIME_STAMP]]
    # # 2. 处理月份和操作类型
    # logger.info("处理月份")
    # part['month'] = part['time_stamp'].apply(month_classify)
    # part = part.drop('time_stamp', axis=1)

    # # 3.合并月份和操作类型 为分类做准备
    # logger.info("合并月份和操作类型")
    # part['act_month'] = pd.Series((repr(x) + "_" + y for x, y in zip(part['action_type'], part['month'])))
    # part = part.drop(['action_type', 'month'], axis=1)

    # # 4. one_hot编码
    # logger.info("one_hot编码")
    # dummies = pd.get_dummies(part[[COL_USER_ID, COL_BRAND_ID, 'act_month']])

    # # 5. 数据透视 求每个月份区间每个商户各种操作数的均值、最大值、中位数、标准差和总数
    # logger.info("数据透视")
    # table = pd.pivot_table(dummies, index=[COL_USER_ID, COL_BRAND_ID], aggfunc=[np.mean, np.max, np.median, np.std, np.sum])

    # # 6. 把aggfunc带来的复合索引化为单层索引
    # logger.info("把aggfunc带来的复合索引化为单层索引")
    # table.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in table.columns]

    # # 7. 输出特征
    # logger.info("输出特征")
    # table = table.reset_index()
    # table.to_csv(GEN_PATH + "feat/u_b/u_b_a_s_feat.csv", index=False)


def u_m_feat(user_log):
    logger = get_log("u_m_feat")

    # 提取用户在每个商户点击商品的总数
    logger.info("提取用户在每个商户点击商品的总数")

    sub = user_log[[COL_USER_ID, COL_MERCHANT_ID, COL_ACTION_TYPE]]

    sub[COL_ACTION_TYPE] = sub[COL_ACTION_TYPE].astype('category')
    user_merchant_feature = pd.get_dummies(sub)

    user_merchant_action_count = pd.pivot_table(user_merchant_feature, index=[COL_USER_ID, COL_MERCHANT_ID],
                                                aggfunc='sum').reset_index()

    user_merchant_action_count.to_csv(GEN_PATH + "feat/u_m/u_m_a_c_feat.csv", index=False)

    # 提取用户提取用户在每个商户点击商品的总数的平均数、标准差、最大值、中位数
    logger.info("提取用户提取用户点击商品的平均数、标准差、最大值、中位数")
    # 提取平均数
    logger.info("合并用户在每个商户点击商品的平均数特征")
    user_merchant_action_avg = pd.pivot_table(user_merchant_action_count, index=[COL_USER_ID, COL_MERCHANT_ID],
                                              values=['action_type_0', 'action_type_1', 'action_type_2',
                                                      'action_type_3'], aggfunc='mean').reset_index().rename(
        columns={'action_type_0': 'action_click_avg', 'action_type_1': 'action_add_to_cart_avg',
                 'action_type_2': 'action_purchase_avg', 'action_type_3': 'action_add_to_favorite_avg'})

    user_merchant_action_avg.to_csv(GEN_PATH + "feat/u_m/u_m_a_avg_feat.csv", index=False)

    logger.info("合并用户在每个商户点击商品的标准差特征")
    user_merchant_action_std = pd.pivot_table(user_merchant_action_count, index=[COL_USER_ID, COL_MERCHANT_ID],
                                              values=['action_type_0', 'action_type_1', 'action_type_2',
                                                      'action_type_3'], aggfunc='std').reset_index().rename(
        columns={'action_type_0': 'action_click_std', 'action_type_1': 'action_add_to_cart_std',
                 'action_type_2': 'action_purchase_std', 'action_type_3': 'action_add_to_favorite_std'})

    user_merchant_action_std.to_csv(GEN_PATH + "feat/u_m/u_m_a_std_feat.csv", index=False)

    logger.info("合并用户在每个商户点击商品的最大值特征")
    user_merchant_action_max = pd.pivot_table(user_merchant_action_count, index=[COL_USER_ID, COL_MERCHANT_ID],
                                              values=['action_type_0', 'action_type_1', 'action_type_2',
                                                      'action_type_3'], aggfunc='max').reset_index().rename(
        columns={'action_type_0': 'action_click_max', 'action_type_1': 'action_add_to_cart_max',
                 'action_type_2': 'action_purchase_max', 'action_type_3': 'action_add_to_favorite_max'})

    user_merchant_action_max.to_csv(GEN_PATH + "feat/u_m/u_m_a_max_feat.csv", index=False)

    logger.info("合并用户在每个商户点击商品的中位数特征")
    user_merchant_action_median = pd.pivot_table(user_merchant_action_count, index=[COL_USER_ID, COL_MERCHANT_ID],
                                                 values=['action_type_0', 'action_type_1', 'action_type_2',
                                                         'action_type_3'], aggfunc='max').reset_index().rename(
        columns={'action_type_0': 'action_click_median', 'action_type_1': 'action_add_to_cart_median',
                 'action_type_2': 'action_purchase_median', 'action_type_3': 'action_add_to_favorite_median'})

    user_merchant_action_median.to_csv(GEN_PATH + "feat/u_m/u_m_a_med_feat.csv", index=False)


def u_c_feat(user_log):
    logger = get_log("u_c_feat")
    sub = user_log[[COL_USER_ID, COL_CAT_ID, COL_ACTION_TYPE]]

    logger.info("提取用户对每个品牌的点击特征")
    sub[COL_ACTION_TYPE] = sub[COL_ACTION_TYPE].astype('category')
    u_c_f = pd.get_dummies(sub)

    u_c_a_c = pd.pivot_table(u_c_f, index=[COL_USER_ID, COL_CAT_ID], aggfunc='sum').reset_index()

    u_c_a_c.to_csv(GEN_PATH + "feat/u_c/u_c_a_c_feat.csv", index=False)

    # 特征太大 注释
    # logger.info("提取用户每个月对每个品牌的点击特征")
    # part = user_log[[COL_USER_ID, COL_CAT_ID, COL_ACTION_TYPE, COL_TIME_STAMP]]
    # # 2. 处理月份和操作类型
    # logger.info("处理月份")
    # part['month'] = part['time_stamp'].apply(month_classify)
    # part = part.drop('time_stamp', axis=1)

    # # 3.合并月份和操作类型 为分类做准备
    # logger.info("合并月份和操作类型")
    # part['act_month'] = pd.Series((repr(x) + "_" + y for x, y in zip(part['action_type'], part['month'])))
    # part = part.drop(['action_type', 'month'], axis=1)

    # # 4. one_hot编码
    # logger.info("one_hot编码")
    # dummies = pd.get_dummies(part[[COL_USER_ID, COL_CAT_ID, 'act_month']])

    # # 5. 数据透视 求每个月份区间每个商户各种操作数的均值、最大值、中位数、标准差和总数
    # logger.info("数据透视")
    # table = pd.pivot_table(dummies, index=[COL_USER_ID, COL_CAT_ID],
    #                        aggfunc=[np.mean, np.max, np.median, np.std, np.sum])

    # # 6. 把aggfunc带来的复合索引化为单层索引
    # logger.info("把aggfunc带来的复合索引化为单层索引")
    # table.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in table.columns]

    # # 7. 输出特征
    # logger.info("输出特征")
    # table = table.reset_index()
    # table.to_csv(GEN_PATH + "feat/u_c/u_c_a_s_feat.csv", index=False)

def m_feat(user_log):
    logger = get_log("m_feat")
    logger.info("提取商户数据集点击/加入购物车/购买/加入到收藏计数特征...")

    sub = user_log[[COL_MERCHANT_ID, COL_ACTION_TYPE]]
    sub[COL_ACTION_TYPE] = sub[COL_ACTION_TYPE].astype('category')

    merchant_count_feature = pd.get_dummies(sub)
    merchant_action_count_feature = pd.pivot_table(merchant_count_feature, index=[COL_MERCHANT_ID],
                                                   aggfunc='sum').reset_index().rename(
        columns={'action_type_0': 'merchant_click_count', 'action_type_1': 'merchant_add_to_cart_count',
                 'action_type_2': 'merchant_purchase_count', 'action_type_3': 'merchant_add_to_favorite_count'})

    logger.info('提取商户点击率/加入购物车率/购买率/加入到收藏计数特征率...')

    merchant_action_count_feature['sum_count'] = merchant_action_count_feature.iloc[:, 1:].apply(sum, axis=1)

    merchant_action_count_feature['merchant_click_rate'] = merchant_action_count_feature['merchant_click_count'] / \
                                                           merchant_action_count_feature['sum_count']
    merchant_action_count_feature['merchant_add_to_cart_rate'] = merchant_action_count_feature[
                                                                     'merchant_add_to_cart_count'] / \
                                                                 merchant_action_count_feature['sum_count']
    merchant_action_count_feature['merchant_purchase_rate'] = merchant_action_count_feature['merchant_purchase_count'] / \
                                                              merchant_action_count_feature['sum_count']
    merchant_action_count_feature['merchant_add_to_favorite_rate'] = merchant_action_count_feature[
                                                                         'merchant_add_to_favorite_count'] / \
                                                                     merchant_action_count_feature['sum_count']

    merchant_action_count_feature.to_csv(GEN_PATH + "feat/m/m_a_c_feat.csv", index=False)

    logger.info('提取商户每月点击/加入购物车/购买/加入到收藏计数特征...')
    part = user_log[[COL_MERCHANT_ID, COL_ACTION_TYPE, COL_TIME_STAMP]]

    # 2. 处理月份和操作类型
    logger.info("处理月份")
    part['month'] = part['time_stamp'].apply(month_classify)
    part = part.drop('time_stamp', axis=1)

    # 3.合并月份和操作类型 为分类做准备
    logger.info("合并月份和操作类型")
    part['act_month'] = pd.Series((repr(x) + "_" + y for x, y in zip(part['action_type'], part['month'])))
    part = part.drop(['action_type', 'month'], axis=1)

    # 4. one_hot编码
    logger.info("one_hot编码")
    dummies = pd.get_dummies(part[[COL_MERCHANT_ID, 'act_month']])

    # 5. 数据透视 求每个月份区间每个商户各种操作数的均值、最大值、中位数、标准差和总数
    logger.info("数据透视")
    table = pd.pivot_table(dummies, index=COL_MERCHANT_ID, aggfunc=[np.mean, np.max, np.median, np.std, np.sum])

    # 6. 把aggfunc带来的复合索引化为单层索引
    logger.info("把aggfunc带来的复合索引化为单层索引")
    table.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in table.columns]

    # 7. 输出特征
    logger.info("输出特征")
    table = table.reset_index()
    table.to_csv(GEN_PATH + "feat/m/m_a_s_feat.csv", index=False)


def m_b_feat(user_log):
    logger = get_log("m_b_feat")
    sub = user_log[[COL_MERCHANT_ID, COL_BRAND_ID, COL_ACTION_TYPE]]

    logger.info("提取用户对每个品牌的点击特征")
    sub[COL_ACTION_TYPE] = sub[COL_ACTION_TYPE].astype('category')
    m_b_f = pd.get_dummies(sub)

    m_b_a_c = pd.pivot_table(m_b_f, index=[COL_MERCHANT_ID, COL_BRAND_ID], aggfunc='sum').reset_index()

    m_b_a_c.to_csv(GEN_PATH + "feat/m_b/m_b_a_c_feat.csv", index=False)

    logger.info("提取用户每个月对每个品牌的点击特征")
    part = user_log[[COL_MERCHANT_ID, COL_BRAND_ID, COL_ACTION_TYPE, COL_TIME_STAMP]]
    # 2. 处理月份和操作类型
    logger.info("处理月份")
    part['month'] = part['time_stamp'].apply(month_classify)
    part = part.drop('time_stamp', axis=1)

    # 3.合并月份和操作类型 为分类做准备
    logger.info("合并月份和操作类型")
    part['act_month'] = pd.Series((repr(x) + "_" + y for x, y in zip(part['action_type'], part['month'])))
    part = part.drop(['action_type', 'month'], axis=1)

    # 4. one_hot编码
    logger.info("one_hot编码")
    dummies = pd.get_dummies(part[[COL_MERCHANT_ID, COL_BRAND_ID, 'act_month']])

    # 5. 数据透视 求每个月份区间每个商户各种操作数的均值、最大值、中位数、标准差和总数
    logger.info("数据透视")
    table = pd.pivot_table(dummies, index=[COL_MERCHANT_ID, COL_BRAND_ID],
                           aggfunc=[np.mean, np.max, np.median, np.std, np.sum])

    # 6. 把aggfunc带来的复合索引化为单层索引
    logger.info("把aggfunc带来的复合索引化为单层索引")
    table.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in table.columns]

    # 7. 输出特征
    logger.info("输出特征")
    table = table.reset_index()
    table.to_csv(GEN_PATH + "feat/m_b/m_b_a_s_feat.csv", index=False)


def i_feat(user_log):
    pass


def i_c_feat(user_log):
    pass


def b_feat(user_log):
    pass


def extract_feature(user_log):
    # 用户特征
    u_feat(user_log)

    # 用户品牌特征
    u_b_feat(user_log)

    # 用户商户特征
    u_m_feat(user_log)

    # 用户类别特征
    u_c_feat(user_log)

    # 商户特征
    m_feat(user_log)

    # 商户品牌特征
    m_b_feat(user_log)

    # 商品特征
    i_feat(user_log)

    # 商品类别特征
    i_c_feat(user_log)

    # 品牌特征
    b_feat(user_log)


def merge_feature(trainset, validset, testset):
    logger = get_log("merge_feature")

    logger.info("合并商户特征")
    for feat in Path(GEN_PATH + "feat/m").glob("*.csv"):
        f = pd.read_csv(feat)
        trainset = trainset.merge(f, on=COL_MERCHANT_ID, how='left', validate='many_to_one')
        validset = validset.merge(f, on=COL_MERCHANT_ID, how='left', validate='many_to_one')
        testset = testset.merge(f, on=COL_MERCHANT_ID, how='left', validate='many_to_one')

    logger.info("合并商户品牌特征")
    for feat in Path(GEN_PATH + "feat/m_b").glob("*.csv"):
        f = pd.read_csv(feat)
        trainset = trainset.merge(f, on=[COL_MERCHANT_ID, COL_BRAND_ID], how='left', validate='many_to_one')
        validset = validset.merge(f, on=[COL_MERCHANT_ID, COL_BRAND_ID], how='left', validate='many_to_one')
        testset = testset.merge(f, on=[COL_MERCHANT_ID, COL_BRAND_ID], how='left', validate='many_to_one')
    
    logger.info("合并用户特征")
    for feat in Path(GEN_PATH + "feat/u").glob("*.csv"):
        f = pd.read_csv(feat)
        trainset = trainset.merge(f, on=COL_USER_ID, how='left', validate='many_to_one')
        validset = validset.merge(f, on=COL_USER_ID, how='left', validate='many_to_one')
        testset = testset.merge(f, on=COL_USER_ID, how='left', validate='many_to_one')
    
    logger.info("合并用户商户特征")
    for feat in Path(GEN_PATH + "feat/u_m").glob("*.csv"):
        f = pd.read_csv(feat)
        trainset = trainset.merge(f, on=[COL_USER_ID, COL_MERCHANT_ID], how='left', validate='many_to_one')
        validset = validset.merge(f, on=[COL_USER_ID, COL_MERCHANT_ID], how='left', validate='many_to_one')
        testset = testset.merge(f, on=[COL_USER_ID, COL_MERCHANT_ID], how='left', validate='many_to_one')

    logger.info("合并用户品牌特征")
    for feat in Path(GEN_PATH + "feat/u_b").glob("*.csv"):
        f = pd.read_csv(feat)
        trainset = trainset.merge(f, on=[COL_USER_ID, COL_BRAND_ID], how='left', validate='many_to_one')
        validset = validset.merge(f, on=[COL_USER_ID, COL_BRAND_ID], how='left', validate='many_to_one')
        testset = testset.merge(f, on=[COL_USER_ID, COL_BRAND_ID], how='left', validate='many_to_one')

    logger.info("合并用户类别特征")
    for feat in Path(GEN_PATH + "feat/u_c").glob("*.csv"):
        f = pd.read_csv(feat)
        trainset = trainset.merge(f, on=[COL_USER_ID, COL_CAT_ID], how='left', validate='many_to_one')
        validset = validset.merge(f, on=[COL_USER_ID, COL_CAT_ID], how='left', validate='many_to_one')
        testset = testset.merge(f, on=[COL_USER_ID, COL_CAT_ID], how='left', validate='many_to_one')

    logger.info("输出训练集...")
    trainset.to_csv(OUTPUT_PATH + "train_feat.csv", index=False)
    logger.info("输出验证集...")
    validset.to_csv(OUTPUT_PATH + "valid_feat.csv", index=False)
    logger.info("输出测试集...")
    testset.to_csv(OUTPUT_PATH + "test_feat.csv", index=False)


def main():
    logger = get_log("get_feature.main")

    # 1. 数据加载
    logger.info("数据加载")
    # # 加载数据
    user_log_format = pd.read_csv(USER_LOG_PATH)

    trainset = pd.read_csv(GEN_PATH + "ex_trainset.csv")
    valiset = pd.read_csv(GEN_PATH + "ex_validset.csv")
    testset = pd.read_csv(EX_TEST_FORMAT)

    # # 2.提取特征
    # logger.info("提取特征")
    extract_feature(user_log_format)

    # 3.合并特征
    logger.info("合并特征并输出")
    merge_feature(trainset, valiset, testset)

    # 特征提取结束
    logger.info("特征提取结束")


if __name__ == '__main__':
    main()
