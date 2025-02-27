import pandas as pd
import pickle
import os

def load_or_create_pkl(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = pd.DataFrame(columns=["dataset"])
    return data

def save_pkl(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def update_dataset(file_path, dataset_name, h, rmse, mae, r2):
    metrics = [rmse, mae, r2]
    
    df = load_or_create_pkl(file_path)
    
    if str(h) not in df.columns:
        df[str(h)] = None
    
    if dataset_name not in df["dataset"].values:
        new_row = {"dataset": dataset_name, str(h): metrics}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        dataset_index = df[df["dataset"] == dataset_name].index[0]
        df.at[dataset_index, str(h)] = metrics
    
    save_pkl(file_path, df)
    

def convert_csv(file_name):
    df = pd.read_pickle(file_name+'.pkl')
    df.to_csv(file_name+'.csv')
    


def split_metrics_tables(file_name):
    df = load_or_create_pkl(file_name+'.pkl')
    
    rmse_df = df.copy()
    mae_df = df.copy()
    r2_df = df.copy()
    
    for col in df.columns[1:]:  # 遍历所有 h 列
        rmse_df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else None)
        mae_df[col] = df[col].apply(lambda x: x[1] if isinstance(x, list) else None)
        r2_df[col] = df[col].apply(lambda x: x[2] if isinstance(x, list) else None)
        
    save_pkl(file_name+'_rmse.pkl', rmse_df)
    save_pkl(file_name+'_mae.pkl', mae_df)
    save_pkl(file_name+'_r2.pkl', r2_df)
    
    return rmse_df, mae_df, r2_df

# split_metrics_tables('ours_4_FFNN')
# convert_csv('ours_4_FFNN_rmse')
# convert_csv('ours_4_FFNN_mae')
# convert_csv('ours_4_FFNN_r2')
# convert_csv('ours_4_FFNN')

