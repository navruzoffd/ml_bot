import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
import re
import torch.nn as nn
import tqdm
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
import scipy.spatial.distance as ds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from catboost import Pool, CatBoostClassifier


def process_json(file):
    # Открываем JSON файл для чтения с указанием кодировки
    with open(file, encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Преобразуем данные из JSON в DataFrame
    df = pd.DataFrame(data)

    # Распаковываем столбец 'docs' в отдельные столбцы 'name', 'description' и 'genres'
    df[['name', 'description', 'genres']] = df['docs'].apply(
        lambda x: pd.Series([x['name'], x['description'], ', '.join([genre['name'] for genre in x['genres']])]))

    # Удаляем столбец 'docs', так как мы уже распаковали его содержимое
    df = df.drop(columns=['docs'])

    # Сохраняем данные в CSV файл
    df.to_csv('data.csv', index=False)

    return df

# Получаем путь к файлу examples.json
file_path = 'examples.json'

# Пример использования функции
df = process_json(file_path)


#Предобработка датафрейма
df['description'] = df['description'].str.lower()

df['description'] = df['description'].str.replace('[{}]'.format(string.punctuation), '')

# Токенизация
df['description'] = df['description'].apply(word_tokenize)

# Удаление стоп слов
stop_words_russian = set(stopwords.words('russian'))
df['description'] = df['description'].apply(lambda x: [word for word in x if word not in stop_words_russian])

# Приведение слов к их изначальной форме
morph = pymorphy2.MorphAnalyzer()
df['description'] = df['description'].apply(lambda x: [morph.parse(word)[0].normal_form for word in x])



# Создаем списки для хранения описания и жанров
max_length = df['description'].str.len().max()
X = []
y = []

# Проходим по каждому фильму в датафрейме
for index, row in df.iterrows():
    description = row['description']
    genre = row['genres'].lower().replace(' ', '').split(',')

    # Добавляем нули в конец описания для уравнивания длины
    description_padded = description + [''] * (max_length - len(description))

    X.append(description_padded)
    y.append(genre)

X = np.asarray(X, dtype="object")
y = np.asarray(y, dtype="object")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = [i[0] for i in y_train]

#загрузка catboost
model1 = CatBoostClassifier()
model1 = CatBoostClassifier().load_model("catboost_model")

cat_features = list(range(0, len(X[0])))

print(len(X), len(cat_features),len(y))

train_dataset = Pool(data=X_train,
                     label=y_train,
                     cat_features=cat_features)

eval_dataset = Pool(data=X_test,
                    label=y_test,
                    cat_features=cat_features)

preds_class = model1.predict(eval_dataset)

preds_proba = model1.predict_proba(eval_dataset)

preds_raw = model1.predict(eval_dataset,
                          prediction_type='RawFormulaVal')

#предсказание жанров
predicted_genres = model1.predict(X)

acc = 0

for i in range(len(preds_class)):
  if (preds_class[i][0] in set(y_test[i])):
    acc+=1

print(acc/len(y_test))

X_array = X.tolist()

X_array = [[item for item in sublist if item != ''] for sublist in X_array]

mod_descr = []
i = 0  # Initialize i before the loop
for index, row in df.iterrows():
    genres = row['genres']
    name = row['name']
    description_str = ' '.join(X_array[i])  # Join inner list into a single string

    # Create a combined list with original structure
    combined_list = []
    for word in description_str.split():
        combined_list.append("'" + word + "'")  # Добавляем каждое слово в формате строки

    # Add genre and name at the end
    combined_list.append("'" + genres + "'")  # Добавляем жанр в формате строки
    combined_list.append("'" + name + "'")  # Добавляем название фильма в формате строки

    # Convert the list back to a string with the original structure
    combined_string = ', '.join(combined_list)  # Объединяем слова запятыми
    mod_descr.append(combined_string)
    i += 1  # Increment i for next X_array element

# Создание нового списка для хранения разделенных слов
split_mod_descr = []

# Проход по каждому элементу в mod_descr1
for element in mod_descr:
    # Разделение строки по запятым и очистка каждого слова от кавычек и пробелов
    split_words = [word.strip().strip("'") for word in element.split(',')]

    while("" in split_words):
      split_words.remove("")

    # Добавление разделенных слов в новый список
    if (split_words!=''):
      split_mod_descr.append(split_words)

samples_X = []
samples_y = []

def slice(arr):
  tek_len = 0
  len_of_slice = 20
  while(tek_len+len_of_slice < len(arr)):
    samples_X.append(arr[tek_len:tek_len+len_of_slice-1])
    samples_y.append(arr[tek_len+len_of_slice-1])
    tek_len += 1

for i in range(len(split_mod_descr)):
  slice(split_mod_descr[i])


#Токенизация данных для обучения

sentences = samples_X + samples_y

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

max_seq_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

word_index = tokenizer.word_index
print("Word index:", word_index)

print("Encoded sequences:")
result = (padded_sequences - padded_sequences.mean()) / padded_sequences.std()

#data_for_train_test

len_X = len(samples_X)


result1 = result[:len(result)-len_X]
result2 = result[len(result)-len_X :]
res = []
for i in result2:
  res.append(i[0])

result2 = res

X = result1
y = result2
X_train, X_validation, y_train, y_validation = train_test_split(X, y, random_state=43, test_size=0.2)

#EarlyStopping


device = 'cpu'
print(f"Using device: {device}")

class EarlyStopping():

  def __init__(self, patience=5, min_delta=1e-3, restore_best_weights=True):
    self.patience = patience
    self.min_delta = min_delta
    self.restore_best_weights = restore_best_weights
    self.best_model = None
    self.best_loss = None
    self.counter = 0
    self.status = ""


  def __call__(self, model, val_loss):
    if self.best_loss == None:
      self.best_loss = val_loss
      self.best_model = copy.deepcopy(model)
    elif self.best_loss - val_loss > self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
      self.best_model.load_state_dict(model.state_dict())
    elif self.best_loss - val_loss < self.min_delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.status = f"Stopped on {self.counter}"
        if self.restore_best_weights:
          model.load_state_dict(self.best_model.state_dict())
        return True


    self.status = f"{self.counter}/{self.patience}"
    return False

#model


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        return out, hn, cn


X_train = torch.Tensor(X_train).float()
y_train = torch.Tensor(y_train).float()

X_validation = torch.Tensor(X_validation).float().to(device)
y_validation = torch.Tensor(y_validation).float().to(device)

batch_size = 16

dataset_train = TensorDataset(X_train, y_train)
dataloader_train = DataLoader(dataset_train,\
  batch_size=batch_size)

dataset_test = TensorDataset(X_validation, y_validation)
dataloader_test = DataLoader(dataset_test,\
  batch_size=batch_size)

model = Net(X.shape[1], 4, 2, 1).to(device)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

es = EarlyStopping()

epoch = 0
done = False
while epoch < 100 and not done:
  epoch += 1
  steps = list(enumerate(dataloader_train))
  pbar = tqdm.tqdm(steps)
  model.train()
  for i, (x_batch, y_batch) in pbar:
    y_batch_pred = model(x_batch.to(device))[0].flatten()
    loss = loss_fn(y_batch_pred[0], y_batch.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss, current = loss.item(), (i + 1)* len(x_batch)
    if i == len(steps)-1:
      model.eval()
      pred = model(X_validation)[0].flatten()
      vloss = loss_fn(pred[0], y_validation)
      if es(model,vloss): done = True
      pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, EStop:[{es.status}]")
    else:
      pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")

def predict(model, input):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
    return predictions

def convert_to_2D(arr):

  arr1 = np.zeros((1, len(arr)))
  for i in range(0, len(arr)):
      arr1[0][i] = arr[i]
  return arr1


def preprocess_user_message(user_message):

  # Преобразовать текст в строчный регистр
  user_message = user_message.lower()  # Apply directly to the string

  # Удалить пунктуацию
  user_message = re.sub('[{}]'.format(string.punctuation), '', user_message)  # Use regular expression

  # Токенизировать текст
  words = word_tokenize(user_message)

  # Удалить стоп-слова
  stop_words_russian = set(stopwords.words('russian'))
  words = [word for word in words if word not in stop_words_russian]

  # Привести слова к нормальной форме
  morph = pymorphy2.MorphAnalyzer()
  normalized_words = [morph.parse(word)[0].normal_form for word in words]


  return normalized_words


# функция добавления жанра в сообщение пользователя
def add_predicted_genre(preprocess_user_message, model):

  # Преобразовать сообщение пользователя в формат, который принимает модель
    #preprocess_user_message_encoded = encoder.transform([preprocess_user_message])

    # Предсказать жанр
    predicted_genre_class = model1.predict(preprocess_user_message+['']*(181-len(preprocess_user_message)))
    preprocess_user_message.append(predicted_genre_class[0])
    # Добавить предсказанный жанр к сообщению
    message_with_genre = preprocess_user_message

    return message_with_genre


samples_df = [] #обрежем описания фильмов до размера 20

def slice(arr):
  len_of_slice = 20
  samples_df.append(arr[0:len_of_slice-1])

for i in range(len(split_mod_descr)):
  slice(split_mod_descr[i])

#Токенизируем сэмплы
sentences = samples_df

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

max_seq_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

result = (padded_sequences - padded_sequences.mean()) / padded_sequences.std()

tensors_df = [] #векторные представления описаний фильмов


for i in range(len(result)):
  out, hidden_states, cell_states = predict(model, torch.Tensor(convert_to_2D(result[i])).float())
  tensors_df.append(hidden_states.numpy())

#добавим индексацию для получения ответов
new_df = df.assign(index=list(range(1, 1001)))

def generate_movies(user_message):
    
    preproc_user_message = preprocess_user_message(user_message)

    predicted_message = add_predicted_genre(preproc_user_message, model)

    tokenized_user_message = tokenizer.texts_to_sequences([predicted_message])
    padded_tokenized_user_message = pad_sequences(tokenized_user_message, maxlen=max_seq_length, padding='post')

    input_arr = convert_to_2D(padded_tokenized_user_message[0])
    out, hidden_states_user, cell_states = predict(model, torch.Tensor(input_arr).float())

    cos_dist = []
    ind = 1
    for i in tensors_df:
        cos_dist.append([ds.cosine(hidden_states_user.flatten(), i.flatten()), ind])
        ind += 1
    cos_dist.sort()
    top5 = cos_dist[:5]
    top5_index = [i[1] for i in top5]

    res_name = new_df[new_df.index.isin(top5_index)]
    res_name = res_name['name'].tolist()
    return res_name