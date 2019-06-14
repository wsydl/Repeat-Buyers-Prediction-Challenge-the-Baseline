import pandas as pd
from random import random
from constant import *


def choose_item(s):
    """
        选取用户在商户双十一购买最多的产品信息
        如果有重复购买则取购买次数最多的产品
    """
    logger.debug(f"处理用户-商户对 => {(s.iloc[0][COL_USER_ID], s.iloc[0][COL_MERCHANT_ID])}")

    if len(s) > 1:
        full = {}
        freq = {}
        for index, row in s.iterrows():
            item_id = row[COL_ITEM_ID]
            if item_id not in full:
                full[item_id] = index

            if item_id not in freq:
                freq[item_id] = 0
            freq[item_id] += 1

        return s.loc[[full[max(freq)]], :]
    return s


def extend_raw(train_format, test_format, user_info, user_log):
    logger = get_log("data_split.extend_raw")

    # 时间处理
    user_log[COL_TIME_STAMP] = user_log[COL_TIME_STAMP].apply(
        lambda x: ('0' + str(x))[-4:])

    # 双十一消费记录
    logger.info('提取双十一消费记录')
    buy_log = user_log[(user_log[COL_ACTION_TYPE] == 2) & (user_log[COL_TIME_STAMP] == '1111')].reset_index(
        drop=True).drop([COL_ACTION_TYPE, COL_TIME_STAMP], 1)

    # 提取(用户, 商户)在双十一购买次数最多的商品信息
    logger.info('提取(用户, 商户)在双十一购买次数最多的商品信息')
    choosen = buy_log.groupby([COL_USER_ID, COL_MERCHANT_ID]).apply(choose_item).reset_index(drop=True)

    # 合并用户特征
    logger.info("合并训练集用户特征")
    train_format = train_format.merge(user_info, on=COL_USER_ID, how='left')
    logger.info("合并测试集用户特征")
    test_format = test_format.merge(user_info, on=COL_USER_ID, how='left')

    # 合并消费记录
    logger.info("合并训练集消费记录")
    extend_train_format = train_format.merge(choosen, on=[COL_USER_ID, COL_MERCHANT_ID], how='left')
    logger.info("合并测试集集消费记录")
    extend_test_format = test_format.merge(choosen, on=[COL_USER_ID, COL_MERCHANT_ID], how='left')

    # 将label移至最后一列
    logger.info("训练集label列移至最后")
    extend_train_format[COL_LABEL] = extend_train_format.pop(COL_LABEL)
    logger.info("测试集label列移至最后")
    extend_test_format[COL_PROB] = extend_test_format.pop(COL_PROB)

    # 输出扩展集
    logger.info("输出扩展后的训练集")
    extend_train_format.to_csv(EX_TRAIN_FORMAT, index=False)
    logger.info("输出扩展后的测试集")
    extend_test_format.to_csv(EX_TEST_FORMAT, index=False)

    return extend_train_format, extend_test_format


if __name__ == '__main__':
    logger = get_log("data_split.main")

    # 1. 数据加载
    logger.info("数据加载")
    train_format = pd.read_csv(TRAIN_FORMAT_PATH)
    test_format = pd.read_csv(TEST_FORMAT_PATH)
    user_info = pd.read_csv(USER_INFO_PATH)
    user_log = pd.read_csv(USER_LOG_PATH)

    # 2. 扩展原始训练集和测试集
    logger.info("扩展原始训练集和测试集")
    train_format, test_format = extend_raw(train_format, test_format, user_info, user_log)

    # 3. 划分训练集和验证集
    logger.info("划分训练集")
    train_format[pd.Series(random() > 0.2 for _ in range(len(train_format)))].to_csv(
        GEN_PATH + "ex_trainset.csv", index=False)
    logger.info("划分验证集")
    train_format[pd.Series(random() < 0.2 for _ in range(len(train_format)))].to_csv(
        GEN_PATH + "ex_validset.csv", index=False)

    logger.info("数据划分结束")