import lightgbm as lgb
from constant import *


def lgb_model(train_path, test_path, valid_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_valid = pd.read_csv(valid_path)

    test_x = df_test.drop([COL_PROB, COL_USER_ID, COL_ITEM_ID, COL_CAT_ID, COL_MERCHANT_ID], axis=1)

    train_y = df_train[COL_LABEL]
    train_x = df_train.drop([COL_LABEL, COL_USER_ID, COL_ITEM_ID, COL_CAT_ID, COL_MERCHANT_ID], axis=1)

    valid_y = df_valid[COL_LABEL]
    valid_x = df_valid.drop([COL_LABEL, COL_USER_ID, COL_ITEM_ID, COL_CAT_ID, COL_MERCHANT_ID], axis=1)
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'objective': 'binary',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1500,
                    valid_sets=lgb_eval)
    print('Save model...')
    # save model to file
    # gbm.save_model('lightgbm/model.txt')
    print('Start predicting...')
    # predict
    pre = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    # eval
    df_test = pd.read_csv(GEN_PATH + "test_raw.csv")
    df_test = df_test[[COL_USER_ID, COL_MERCHANT_ID]]
    df_test['prob'] = pd.Series(pre)
    df_test[[COL_USER_ID, COL_MERCHANT_ID, 'prob']].to_csv('../output/only_user_merchant/prediction.csv', index=False)

def main():
    file_train = OUTPUT_PATH + "/feat_only_user_merchant/train_feat_only_user_merchant.csv"
    file_test = OUTPUT_PATH + "/feat_only_user_merchant/test_feat_only_user_merchant.csv"
    file_valid = OUTPUT_PATH + "/feat_only_user_merchant/valid_feat_only_user_merchant.csv"
    lgb_model(file_train, file_test, file_valid)


if __name__ == '__main__':
    main()
