# 使用xgb训练模型

from constant import *
import time
import xgboost as xgb
import numpy as np

def main():
    # 记录程序运行时间
    start_time = time.time()
    logger = get_log("xgb_model.main")

    # 读取数据
    logger.info("数据加载")
    train = pd.read_csv(GEN_PATH + "train_feat.csv")
    test = pd.read_csv(GEN_PATH + "valid_feat.csv")
    pre = pd.read_csv(GEN_PATH + "test_feat.csv")
    pre = pre.drop([COL_PROB], axis=1)

    train_label = train[[COL_LABEL]]
    train_data = train.drop([COL_LABEL], axis=1)

    test_label = test[[COL_LABEL]]
    test_data = test.drop([COL_LABEL], axis=1)

    xgb_val = xgb.DMatrix(test_data, label=test_label)
    xgb_train = xgb.DMatrix(train_data, label=train_label)
    xgb_pre = xgb.DMatrix(pre)

    # xgboost模型
    # 设置参数
    logger.info("设置参数")
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',      # 二分类的问题，返回预测概率
        'gamma': 0.1,               # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子
        'max_depth': 8,            # 构建输的深度，越大越容易过拟合   12
        'lambda': 4,                # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.8,           # 随机采样训练样本
        'colsample_bytree': 1,    # 生成树时进行的列采样
        'min_child_weight': 5,      # 3
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.02,  # 如同学习率
        'seed': 100,
        # 'nthread': 8,  # cpu 线程数
        'eval_metric': 'auc'
    }
    # plst = list(params.items())
    num_rounds = 200   # 迭代次数
    watchlist = [(xgb_train, 'train', xgb_val, 'val')]

    # 训练模型并保存
    logger.info("训练模型")
    model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    model.save_model('../model/xgb.model')  # 用于存储训练出的模型
    print("best best_ntree_limit", model.best_ntree_limit)

    # 预测并保存
    logger.info("预测并保存")
    preds = model.predict(xgb_pre, ntree_limit=model.best_ntree_limit)

    # np.savetxt('xgb_submission.csv', np.c_[range(1, len(pre) + 1), preds], delimiter=',',
    #            header='user_id, merchant_id, prob',
    #            comments='', fmt='%d')
    pre = pd.read_csv(GEN_PATH + "test_raw.csv")
    pre = pre[[COL_USER_ID, COL_MERCHANT_ID]]
    pre['prob'] = pd.Series(preds)
    pre[[COL_USER_ID, COL_MERCHANT_ID, 'prob']].to_csv('../output/prediction.csv', index=False)

    # 输出运行时长
    cost_time = time.time() - start_time
    print("xgboost success!", '\n', "cost time:", cost_time, "(s)......")

if __name__=='__main__':
    main()
