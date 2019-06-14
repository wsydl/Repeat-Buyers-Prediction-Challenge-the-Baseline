import logging
import sys
import pandas as pd

# 路径常量
TRAIN_FORMAT_PATH = '../data/train_format1.csv'
TEST_FORMAT_PATH = '../data/test_format1.csv'
USER_INFO_PATH = '../data/user_info_format1.csv'
USER_LOG_PATH = '../data/user_log.csv'
USER_LOG_SAMPLE_PATH = '../data/user_log_sample.csv'
GEN_PATH = '../gen/'
GEN_USER_LOG_PATH = '../gen/user_log/'
EX_TRAIN_FORMAT = GEN_PATH + "extended_train_format.csv"
EX_TEST_FORMAT = GEN_PATH + "extended_test_format.csv"

OUTPUT_PATH = '../output/'

# 列名常量
COL_USER_ID = 'user_id'
COL_AGE = 'age_range'
COL_GENDER = 'gender'
COL_MERCHANT_ID = 'merchant_id'
COL_LABEL = 'label'
COL_CAT_ID = 'cat_id'
COL_ITEM_ID = 'item_id'
COL_BRAND_ID = 'brand_id'
COL_TIME_STAMP = 'time_stamp'
COL_ACTION_TYPE = 'action_type'
COL_PROB = 'prob'

# 特征集 名称: 关键字
PROFILES = {
    'user': [COL_USER_ID],  # 用户特征
    'merchant': [COL_MERCHANT_ID],  # 商户特征
    'item': [COL_ITEM_ID],  # 项特征
    'brand': [COL_BRAND_ID],  # 品牌特征
    'cat': [COL_CAT_ID],  # 分类特征
    'user_merchant': [COL_USER_ID, COL_MERCHANT_ID],  # 用户-商户特征
    'user_brand': [COL_USER_ID, COL_BRAND_ID],  # 用户-品牌特征
    'user_cat': [COL_USER_ID, COL_CAT_ID],  # 用户-分类特征
    'merchant_brand': [COL_MERCHANT_ID, COL_BRAND_ID],  # 商户-品牌特征
    'merchant_cat': [COL_MERCHANT_ID, COL_CAT_ID]  # 商户-分类特征
}


def get_log(appname, lv=logging.INFO):
    logger = logging.getLogger(appname)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-4s: %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter
    logger.addHandler(console_handler)
    logger.setLevel(lv)

    return logger


def count_action(group):
    months = ['5_6', '6_7', '7_8', '8_9', '9_10', '10_11']
    actions = {
        0: "click",
        1: "add_to_cart",
        2: "purchase",
        3: "add_to_favorite"
    }
    counts = {COL_USER_ID: [group.iat[0, 0]]}
    cols = [COL_USER_ID]
    for month_area in months:
        monthly_actions_counts_total = 0

        for action in actions.keys():
            col_name = f'{month_area}_monthly_{actions[action]}_count'
            count = len(group[(group[COL_ACTION_TYPE] == action) & (group['month'] == month_area)])
            counts[col_name] = pd.Series(count)

            monthly_actions_counts_total += count

        monthly_actions_counts_total = pd.Series(monthly_actions_counts_total)

        for action in actions.keys():
            col_name = f'{month_area}_monthly_{actions[action]}_count'
            rate_col = f'{month_area}_user_monthly_{actions[action]}_rate'
            cols.append(col_name)
            cols.append(rate_col)

            rate = counts[col_name] / monthly_actions_counts_total

            counts[rate_col] = rate

    return pd.DataFrame(counts, columns=cols)


def month_classify(time_stamp):
    if 511 <= time_stamp < 612:
        return "5_6"
    elif 612 <= time_stamp < 712:
        return "6_7"
    elif 712 <= time_stamp < 812:
        return "7_8"
    elif 812 <= time_stamp < 912:
        return "8_9"
    elif 912 <= time_stamp < 1012:
        return "9_10"
    elif 1012 <= time_stamp <= 1112:
        return "10_11"
    else:
        return 0


def calculate_rate(x, month, action_type):
    return pd.DataFrame(x[month + action_type] / (x[month + '_monthly_click_count'] +
                                                  x[month + '_monthly_add_to_cart_count'] +
                                                  x[month + '_monthly_purchase_count'] +
                                                  x[month + '_monthly_add_to_favorite_count']))

