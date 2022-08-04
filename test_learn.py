#データのダウンロード
#!wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz 
#ファイルの解凍
#!tar -zxf ldcc-20140209.tar.gz 

import random
import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
# 日本語の事前学習モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking’

# データローダーの作成
dataset_for_loader = [
    {'data':torch.tensor([0,1]), 'labels':torch.tensor(0)},
    {'data':torch.tensor([2,3]), 'labels':torch.tensor(1)},
    {'data':torch.tensor([4,5]), 'labels':torch.tensor(2)},
    {'data':torch.tensor([6,7]), 'labels':torch.tensor(3)},
]
loader = DataLoader(dataset_for_loader, batch_size=2)
# データセットからミニバッチを取り出す
for idx, batch in enumerate(loader):
    print(f'# batch {idx}')
    print(batch)
    ## ファインチューニングではここでミニバッチ毎の処理を行う

loader = DataLoader(dataset_for_loader, batch_size=2, shuffle=True)
for idx, batch in enumerate(loader):
    print(f'# batch {idx}')
    print(batch)

# カテゴリーのリスト
category_list = [
    'dokujo-tsushin',
    'it-life-hack',
    'kaden-channel',
    'livedoor-homme',
    'movie-enter',
    'peachy',
    'smax',
    'sports-watch',
    'topic-news'
]

# トークナイザのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

# 各データの形式を整える
max_length = 128
dataset_for_loader = []
for label, category in enumerate(tqdm(category_list)):
    for file in glob.glob(f'./text/{category}/{category}*'):
        lines = open(file).read().splitlines()
        text = '\n'.join(lines[3:]) # ファイルの4行目からを抜き出す。
        encoding = tokenizer(
            text,
            max_length=max_length, 
            padding='max_length',
            truncation=True
        )
        encoding['labels'] = label # ラベルを追加
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_for_loader.append(encoding)

print(dataset_for_loader[0])

# データセットの分割
random.shuffle(dataset_for_loader) # ランダムにシャッフル
n = len(dataset_for_loader)
n_train = int(0.6*n)
n_val = int(0.2*n)
dataset_train = dataset_for_loader[:n_train] # 学習データ
dataset_val = dataset_for_loader[n_train:n_train+n_val] # 検証データ
dataset_test = dataset_for_loader[n_train+n_val:] # テストデータ

# データセットからデータローダを作成
# 学習データはshuffle=Trueにする。
dataloader_train = DataLoader(
    dataset_train, batch_size=32, shuffle=True
) 
dataloader_val = DataLoader(dataset_val, batch_size=256)
dataloader_test = DataLoader(dataset_test, batch_size=256)

class BertForSequenceClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率
        super().__init__()
        
        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters() 
        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        return loss
        
    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss) # 損失を'val_loss'の名前でログをとる。
    # テストデータのミニバッチが与えられた時に、
    # テストデータを評価する指標を計算する関数を書く。
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        self.log('accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。
    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# 学習時にモデルの重みを保存する条件を指定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/',
)
# 学習の方法を指定
trainer = pl.Trainer(
    gpus=1, 
    max_epochs=10,
    callbacks = [checkpoint]
)
# PyTorch Lightningモデルのロード
model = BertForSequenceClassification_pl(
    MODEL_NAME, num_labels=9, lr=1e-5
)
# ファインチューニングを行う。
trainer.fit(model, dataloader_train, dataloader_val) 

