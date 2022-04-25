import sys
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import gc
    
    
def main():
    data, output_path = sys.argv[1:]
    transactions = pd.read_csv(f'{data}/transactions.csv')
    transactions['transaction_dttm'] = pd.to_datetime(transactions['transaction_dttm'])
    transactions['hour'] = transactions['transaction_dttm'].dt.hour
    transactions_dtypes = {'mcc_code':np.int16, 'currency_rk':np.int8}
    transactions = transactions.astype(transactions_dtypes)
    bankclient_embed = transactions .pivot_table(index = 'user_id', 
                            values=['transaction_amt'],
                            columns=['mcc_code'],
                            aggfunc=['sum','mean', 'count']).fillna(0)
    bankclient_embed.columns = [f'bank_{str(i[0])}-{str(i[2])}' for i in bankclient_embed.columns]
    
    clickstream = pd.read_csv(f'{data}/clickstream.csv')
    clickstream['timestamp'] = pd.to_datetime(clickstream['timestamp'])
    clickstream['hour'] = clickstream['timestamp'].dt.hour
    clickstream_dtypes = {'cat_id':np.int16, 'new_uid':np.int32}
    clickstream = clickstream.astype(clickstream_dtypes)
    clickstream_embed = clickstream.pivot_table(index = 'user_id', 
                            values=['timestamp'],
                            columns=['cat_id'],
                            aggfunc=['count']).fillna(0)
    clickstream_embed.columns = [f'rtk_{str(i[0])}-{str(i[2])}' for i in clickstream_embed.columns]
    clickstream_embed.loc[0] = np.empty(len(clickstream_embed.columns))

    tr_sv = pd.pivot_table(transactions, index='user_id', columns='hour', values = 'transaction_amt', aggfunc = 'count').fillna(0)
    tr_sv['summs'] = tr_sv.sum(axis=1)
    for i in tr_sv.columns[:-1]:
        tr_sv[i] /= tr_sv['summs']
    tr_sv.columns = ['trans_h_'+ str(i) for i in tr_sv.columns]
    cl_sv = pd.pivot_table(clickstream, index='user_id', columns='hour', values = 'timestamp', aggfunc = 'count').fillna(0)
    cl_sv['summs'] = cl_sv.sum(axis=1)
    for i in cl_sv.columns[:-1]:
        cl_sv[i] /= cl_sv['summs']
    cl_sv.columns = ['click_h_'+ str(i) for i in cl_sv.columns]
    
    del clickstream
    gc.collect()

    list_of_rtk = list(clickstream_embed.index.unique())
    list_of_bank= list(bankclient_embed.index.unique())
    submission = pd.DataFrame(list_of_bank, columns=['bank'])
    submission['rtk'] = submission['bank'].apply(lambda x: list_of_rtk)

    model = CatBoostClassifier()
    model.load_model('open_sol_2504.cbm',  format='cbm') 

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
            part_of_submit = part_of_submit.sort_values(by=['bank', 'predicts'], ascending=False).reset_index(drop=True)

            part_of_submit = part_of_submit.pivot_table(index='bank', values='rtk', aggfunc=list)
            part_of_submit['rtk'] = part_of_submit['rtk'].apply(lambda x: x[:100])
            part_of_submit['bank'] = part_of_submit.index
            part_of_submit = part_of_submit[['bank', 'rtk']]
            submission_ready.extend(part_of_submit.values)
    
    submission_final = np.array(submission_ready, dtype=object)
    np.savez(output_path, submission_final)

if __name__ == "__main__":
    main()