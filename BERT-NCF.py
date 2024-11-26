import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def negative_sampling(user_item_matrix, user_dict, item_dict, neg_ratio=1):

    num_users, num_items = user_item_matrix.shape
    positive_samples = []
    negative_samples = []
    
    for user_id in range(num_users):

        positive_items = torch.where(user_item_matrix[user_id] == 1)[0]

        negative_items = torch.where(user_item_matrix[user_id] == 0)[0]
        
        positive_samples.extend(
            [(user_dict[user_id], item_dict[item_id.item()], 1) for item_id in positive_items]
        )
        
        num_negatives = len(positive_items) * neg_ratio
        sampled_negatives = torch.randint(
            low=0, high=len(negative_items), size=(num_negatives,), generator=None
        )
        negative_samples.extend(
            [(user_dict[user_id], item_dict[negative_items[idx].item()], 0) for idx in sampled_negatives]
        )
    
    all_samples = positive_samples + negative_samples
        
    np.random.shuffle(all_samples)
    
    return all_samples

def full_sampling(user_item_matrix, user_dict, item_dict):
    num_users, num_items = user_item_matrix.shape
    all_samples = []
    id_samples = []
    for user_id in range(num_users):
        for item_id in range(num_items):
            label = user_item_matrix[user_id, item_id].item() 
            user_value = user_dict[user_id]  
            item_value = item_dict[item_id]  
            all_samples.append((user_value, item_value, label))  
            id_samples.append((user_id, item_id))
    return all_samples,id_samples

def process_matrix_data(rating_file, user_file, item_file):
    
    rating_df = pd.read_csv(rating_file)
    user_df = pd.read_csv(user_file)
    item_df = pd.read_csv(item_file)
    
    user_dict = user_df.set_index('user_id')['question'].to_dict()
    item_dict = item_df.set_index('item_id')['symbol'].to_dict()
    
    num_users = user_df['user_id'].nunique()
    num_items = item_df['item_id'].nunique()

    ratings_matrix = torch.zeros((num_users, num_items))

    for i, row in rating_df.iterrows():
        user_idx = row['user_id']
        item_idx = row['item_id']
        rating = row['rating']
        ratings_matrix[user_idx, item_idx] = rating

    return ratings_matrix, user_dict, item_dict , user_df['question'], item_df['symbol']

class RecommendModel(nn.Module):
    def __init__(self):
        super(RecommendModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
        self.encoder = BertModel.from_pretrained('chinese-bert-wwm-ext')
        
        self.MLP_user_projection = nn.Linear(768, 768)
        self.MLP_item_projection = nn.Linear(768, 768)
        self.MF_user_projection = nn.Linear(768, 768)
        self.MF_item_projection = nn.Linear(768, 768)
        
        self.MLP = nn.Sequential(
            nn.Linear(1536,1152),
            nn.ReLU(),
            nn.Linear(1152,768),
            nn.LayerNorm(768),
            nn.ReLU(),
        )
        
        self.NeuMF = nn.Sequential(
            nn.Linear(1536,768),
            nn.ReLU(),
            nn.Linear(768,256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Linear(32, 1)
        )

    def forward(self, user_data, item_data):
        
        user_data_inputs = self.tokenizer(user_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
        user_data_inputs = {key: val.to(device) for key, val in user_data_inputs.items()}
        usr_outputs = self.encoder(**user_data_inputs).last_hidden_state[:, 0, :]
        MLP_usr_outputs = self.MLP_user_projection(usr_outputs)
        MF_usr_outputs = self.MF_user_projection(usr_outputs)        
        
        item_data_inputs = self.tokenizer(item_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
        item_data_inputs = {key: val.to(device) for key, val in item_data_inputs.items()}
        item_outputs = self.encoder(**item_data_inputs).last_hidden_state[:, 0, :]
        MLP_item_outputs = self.MLP_item_projection(item_outputs)
        MF_item_outputs = self.MF_item_projection(item_outputs)
        
        ui_combined = torch.cat([MLP_usr_outputs, MLP_item_outputs], dim=1)
        
        ui_mu = MF_usr_outputs * MF_item_outputs
        
        ui_combined = self.MLP(ui_combined)
        
        combined = torch.cat([ui_combined,ui_mu], dim=1)
        
        predictions = self.NeuMF(combined)
        
        predictions = torch.sigmoid(predictions).squeeze()
        
        return predictions
        

if __name__ == "__main__":
    
    train_ratings_matrix, train_users_dict, train_items_dict, train_user, train_item = process_matrix_data(
    'new_data/train/rating_data.csv',
    'new_data/train/user_data.csv',
    'new_data/item_data.csv'
    )
    
    test_ratings_matrix, test_users_dict, test_items_dict, test_user, test_item = process_matrix_data(
    'new_data/test/rating_data.csv',
    'new_data/test/user_data.csv',
    'new_data/item_data.csv'
    )
    
    train_data = negative_sampling(train_ratings_matrix,train_users_dict, train_items_dict, neg_ratio=1)
    
    test_data, id_samples = full_sampling(test_ratings_matrix, test_users_dict, test_items_dict)
    
    recommend_model = RecommendModel().to(device)
    
    criterion = nn.BCELoss()
    learning_rate_bert = 2e-5
    learning_rate_others = 1e-5
    
    optimizer = optim.Adam([
        {'params': recommend_model.encoder.parameters(), 'lr': learning_rate_bert},
        {'params': recommend_model.MLP_user_projection.parameters()},
        {'params': recommend_model.MLP_item_projection.parameters()},
        {'params': recommend_model.MF_item_projection.parameters()},
        {'params': recommend_model.MF_user_projection.parameters()},   
        {'params': recommend_model.MLP.parameters()},
        {'params': recommend_model.NeuMF.parameters()},
    ], lr=learning_rate_others)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_user, train_item, train_ratings = zip(*train_data)
    train_ratings = torch.tensor(train_ratings).float().to(device)
    
    epochs = 20
    batch_size = 32
    recommend_model.train()

    for epoch in range(epochs):
        
        epoch_loss = 0
        for i in range(0, len(train_user), batch_size):
            optimizer.zero_grad()
            
            user_batch = train_user[i:i + batch_size]
            item_batch = train_item[i:i + batch_size]
            rating_batch = train_ratings[i:i + batch_size]
            
            predictions = recommend_model(user_batch, item_batch)
            # print(predictions)
            loss = criterion(predictions, rating_batch)
            loss.backward()
            optimizer.step()
    
        scheduler.step()
        epoch_loss += loss.item()
    
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (len(train_user) // batch_size)}")


    test_user, test_item, test_ratings = zip(*test_data)
    test_ratings = torch.tensor(test_ratings).float()
    
    batch_size=32
    predictions = []
    recommend_model.eval()
    
    for i in range(0, len(test_user), batch_size):
        user_batch = test_user[i:i + batch_size]
        item_batch = test_item[i:i + batch_size]
        rating_batch = test_ratings[i:i + batch_size]
        with torch.no_grad():
            predictions.extend(recommend_model(user_batch, item_batch).cpu().tolist())
    
    predictions = torch.tensor(predictions)

    loss = criterion(predictions, test_ratings)
    
    print(f"Test Loss: {loss.item()}")
    
    top_n = 10
    top_10_recommendations = {}
    
    predictions_df = pd.DataFrame({
        'user_id': [sample[0] for sample in id_samples],
        'item_id': [sample[1] for sample in id_samples],
        'predicted_rating': predictions.numpy()
    })
    item_df = pd.read_csv('new_data/item_data.csv')
    item_df = item_df.set_index('item_id')
    
    for user_id in predictions_df['user_id'].unique():
        user_predictions = predictions_df[predictions_df['user_id'] == user_id]
    
        top_items = user_predictions.sort_values(by='predicted_rating', ascending=False).head(top_n)
    
        top_items_ids = top_items['item_id'].tolist()
        print(top_items_ids)
        
        item_symbols = {item_id: item_df.loc[item_id, 'symbol'] for item_id in top_items_ids}
    
        item_ratings = {item_id: top_items[top_items['item_id'] == item_id]['predicted_rating'].values[0] for item_id in
                        top_items_ids}
    
        top_10_recommendations[user_id] = {
            'predicted_symbols': [
                {'symbol': item_symbols[item_id], 'probability': item_ratings[item_id]} for item_id in top_items_ids
            ]
        }
    

    with open('new_data/test_data.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    for i, item in enumerate(existing_data):
        try:
            item['recommended_symbols'] = top_10_recommendations[i]['predicted_symbols']
        except:
            print(f"No recommendations for usr {i}, setting to empty list.")
            item['recommended_symbols'] = []
    

    def convert_numpy_float32(obj):
        if isinstance(obj, dict):

            return {k: convert_numpy_float32(v) for k, v in obj.items()}
        elif isinstance(obj, list):

            return [convert_numpy_float32(item) for item in obj]
        elif isinstance(obj, np.float32):

            return float(obj)
        else:

            return obj
    
    

    existing_data = convert_numpy_float32(existing_data)
    
    with open('updated_data_test_origin_2.json', 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
