import sys
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import gc
    

def read_cl_data(input_folder='data', all_dicts = {}):
    clickstream = pd.read_csv(f'{input_folder}/clickstream.csv')
    clickstream['timestamp'] = pd.to_datetime(clickstream['timestamp'])
    all_dicts['rtk_le'] = LabelEncoder().fit(clickstream['user_id'])
    clickstream['user_id'] = all_dicts['rtk_le'].transform(clickstream['user_id'])+1
    clickstream_dtypes = {'user_id':np.int16, 'cat_id':np.int16, 'new_uid':np.int32}
    clickstream = clickstream.astype(clickstream_dtypes)
    return clickstream, all_dicts
    
    
def read_tr_data(input_folder='data', all_dicts = {}):
    transactions = pd.read_csv(f'{input_folder}/transactions.csv')
    transactions['transaction_dttm'] = pd.to_datetime(transactions['transaction_dttm'])
    all_dicts['bank_le'] = LabelEncoder().fit(transactions['user_id'])
    transactions['user_id'] = all_dicts['bank_le'].transform(transactions['user_id'])+1
    transactions_dtypes = {'user_id':np.int16, 'mcc_code':np.int16, 'currency_rk':np.int8}
    transactions = transactions.astype(transactions_dtypes)
    return transactions, all_dicts
    
    
def read_train_data(all_dicts, input_folder='data'):
    train = pd.read_csv(f'{input_folder}/train_matching.csv')
    train['bank'] = all_dicts['bank_le'].transform(train['bank'])+1
    train.loc[train.rtk=='0', 'rtk'] = 0
    train.loc[train.rtk!=0, 'rtk'] = all_dicts['rtk_le'].transform(train.loc[train.rtk!=0, 'rtk'])+1
    return train
    
    
def new_feats(clickstream, time_col, naming):
    clickstream['hour'] = clickstream[time_col].dt.hour
    cl_sv = pd.pivot_table(clickstream, index='user_id', columns='hour', values = time_col, aggfunc = 'count').fillna(0)
    cl_sv['summs'] = cl_sv.sum(axis=1)
    for i in cl_sv.columns[:-1]:
        cl_sv[i] /= cl_sv['summs']
    cl_sv.columns = [f'{naming}_h_'+ str(i) for i in cl_sv.columns]
    return cl_sv
    
    
def get_baseline_embed(clickstream, time_col, cat_col, naming, aggfunc):
    clickstream_embed = clickstream.pivot_table(index = 'user_id', 
                            values=[time_col],
                            columns=[cat_col],
                            aggfunc=aggfunc).fillna(0)
    clickstream_embed.columns = [f'{naming}_{str(i[0])}-{str(i[2])}' for i in clickstream_embed.columns]
    clickstream_embed.loc[0] = np.empty(len(clickstream_embed.columns))
    dtype = pd.SparseDtype(np.int32, fill_value=0)
    clickstream_embed = clickstream_embed.astype(dtype)
    return clickstream_embed
    
    
def main():
    data, output_path = sys.argv[1:]
    # Read data and label encoder to decrease used RAM
    clickstream, all_dicts = read_cl_data(input_folder=data, all_dicts={})
    # New feats basing on hour embeddings
    cl_sv = new_feats(clickstream, 'timestamp', 'click')
    # Embedings on categories from baseline
    clickstream_embed = get_baseline_embed(clickstream, 'timestamp', 'cat_id', 'rtk', aggfunc = ['count'])
    del clickstream
   
    # Read data and label encoder to decrease used RAM
    transactions, all_dicts = read_tr_data(input_folder='data', all_dicts=all_dicts)
    # New feats basing on hour embeddings
    tr_sv = new_feats(transactions, 'transaction_dttm', 'trans')
    # Embedings on categories from baseline
    bankclient_embed = get_baseline_embed(transactions, 'transaction_amt', 'mcc_code', 'bank', aggfunc = ['sum','mean', 'count'])
    del transactions

    list_of_rtk = list(clickstream_embed.index.unique())
    list_of_bank= list(bankclient_embed.index.unique())
    
    submission = pd.DataFrame(list_of_bank, columns=['bank'])
    submission['rtk'] = submission['bank'].apply(lambda x: list_of_rtk)


    model = CatBoostClassifier()
    model.load_model('open_sol_model_1.cbm',  format='cbm')
    
    submission_ready = []
    batch_size = 150
    num_of_batches = int((len(list_of_bank))/batch_size)+1


    for i in range(num_of_batches):
        bank_ids = list_of_bank[(i*batch_size):((i+1)*batch_size)]
        if len(bank_ids) != 0:
            part_of_submit = submission[submission['bank'].isin(bank_ids)].explode('rtk')
            part_of_submit = part_of_submit.merge(bankclient_embed, how='left', left_on='bank', right_index=True
                                        ).merge(clickstream_embed, how='left', left_on='rtk', right_index=True
                                                              ).merge(tr_sv, how='left', left_on='bank', right_index=True
                                                              ).merge(cl_sv, how='left', left_on='rtk', right_index=True
                                                              ).fillna(0)
            for i in model.feature_names_:
                if i not in part_of_submit.columns:
                    part_of_submit[i] = 0

            part_of_submit['predicts'] = model.predict_proba(part_of_submit[model.feature_names_])[:,1]
            
            part_of_submit = part_of_submit[['bank', 'rtk', 'predicts']]

            zeros_part = pd.DataFrame(part_of_submit['bank'].unique(), columns=['bank'])
            zeros_part['rtk'] = 0.
            zeros_part['predicts'] = 3.8
            
            part_of_submit = pd.concat((part_of_submit, zeros_part))
            part_of_submit = part_of_submit.sort_values(by=['bank', 'predicts'], ascending=False).reset_index(drop=True)
            part_of_submit['pred_rank'] = part_of_submit.groupby('bank')['predicts'].rank(ascending=False)
            part_of_submit = part_of_submit[part_of_submit['pred_rank']<=101]
            part_of_submit = part_of_submit.sort_values(by=['bank', 'predictss'], ascending=False).reset_index(drop=True)

            part_of_submit = part_of_submit.pivot_table(index='bank', values='rtk', aggfunc=list)
            part_of_submit['rtk'] = part_of_submit['rtk'].apply(lambda x: x[:100])
            part_of_submit['bank'] = part_of_submit.index
            part_of_submit = part_of_submit[['bank', 'rtk']]
            submission_ready.extend(part_of_submit.values)
    
    submission_final = np.array(submission_ready, dtype=object)

    print(submission_final.shape)
    np.savez(output_path, submission_final)

if __name__ == "__main__":
    main()