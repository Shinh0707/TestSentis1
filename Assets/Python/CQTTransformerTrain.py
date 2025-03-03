import torch
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor
import numpy as np
import librosa
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ChromaTransformer import CQTTransformer

# パラメータ設定
SR = 44100  # サンプルレート
BINS = 84   # CQTのビン数
FMIN = 110  # 最低周波数
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
MAX_SEQUENCES = 1000
LAYERS = 3
SEGMENT_LENGTH = 44100 // 512  # セグメント長
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 音声データセットの定義
class AudioDataset(Dataset):
    def __init__(self, audio_folder, segment_length=5, sr=44100, bins=84, fmin=110):
        self.audio_files = glob.glob(os.path.join(os.path.dirname(__file__),audio_folder, "*.wav"))
        print(self.audio_files)
        self.segment_length = segment_length  # 秒単位
        self.sr = sr
        self.bins = bins
        self.fmin = fmin
        self.segments = self._create_segments()
        
    def _create_segments(self):
        segments = np.empty((0,self.bins))
        for file_path in tqdm(self.audio_files, desc="音声ファイルの読み込み"):
            try:
                # 音声読み込み
                y, _ = librosa.load(file_path, sr=self.sr, mono=True)
                # 現在のセグメントのCQT変換
                max_y = np.max(np.abs(y))
                max_y = 1 if max_y == 0 else max_y
                cqt = np.abs(librosa.cqt(y/max_y, sr=self.sr, fmin=self.fmin, n_bins=self.bins))
                cqt: np.ndarray = cqt.transpose(1,0)
                segments = np.concat([segments,cqt],axis=0)
            except Exception as e:
                print(f"エラー: {file_path} の処理中にエラーが発生しました - {e}")
        
        print(f"合計 {len(segments)} セグメントを作成しました")
        return segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx:idx+self.segment_length+1]
        segment = np.pad(segment, ((0,self.segment_length-len(segment)+1),(0,0)))
        x = FloatTensor(segment[:-1])
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = FloatTensor(segment[-1])
        
        return x, y

# 学習ループ
def train_model(model, train_loader, optimizer, criterion, device, epochs):
    model.to(device)
    model.train()
    
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_losses = []
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"エポック {epoch+1}/{epochs}")):
            x, y = x.to(device), y.to(device)
            
            # 勾配のリセット
            optimizer.zero_grad()
            
            # 順伝播
            output = model(x)
            # 損失計算
            loss = criterion(output, y)
            
            # 逆伝播
            loss.backward()
            
            # パラメータ更新
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_losses.append(loss.item())
            
            # 100バッチごとに進捗表示
            if (batch_idx + 1) % 100 == 0:
                print(f"バッチ {batch_idx+1}/{len(train_loader)}, 損失: {loss.item():.6f}")
        epoch_losses = np.array(epoch_losses)
        avg_epoch_loss = epoch_losses.mean()
        train_losses.append(avg_epoch_loss)
        print(f"エポック {epoch+1}/{epochs} 平均損失: {avg_epoch_loss:.6f} 標準偏差: {epoch_losses.std()}")
        
        # モデルの保存（5エポックごと）
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, os.path.join(os.path.dirname(__file__),f"cqt_transformer_epoch_{epoch+1}.pt"))
    
    # 学習曲線のプロット
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__),'training_loss.png'))
    plt.close()
    
    return train_losses

def main():
    # データセットとデータローダーの準備
    train_dataset = AudioDataset(
        audio_folder="Train",
        segment_length=SEGMENT_LENGTH,
        sr=SR,
        bins=BINS,
        fmin=FMIN
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # モデル、損失関数、オプティマイザの準備
    model = CQTTransformer(
        nbins=BINS,
        layers=LAYERS,
        max_seq_len=MAX_SEQUENCES,
        dropout=0.1
    )
    # モデルの重みをロード
    # model_path = os.path.join(os.path.dirname(__file__), "cqt_transformer_final.pt")  # 実際のパスに変更してください
    # model.load_state_dict(torch.load(model_path)['model_state_dict'])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # トレーニングの実行
    losses = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        epochs=NUM_EPOCHS
    )
    
    # 最終モデルの保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses[-1],
    }, os.path.join(os.path.dirname(__file__),"cqt_transformer_final.pt"))
    
    print("学習完了！")

def toOnnx(model_filename: str, output_filename: str):
    model = CQTTransformer(
        nbins=BINS,
        layers=LAYERS,
        max_seq_len=MAX_SEQUENCES,
        dropout=0
    )
    # モデルの重みをロード
    model_path = os.path.join(os.path.dirname(__file__), model_filename)  # 実際のパスに変更してください
    out_path = os.path.join(os.path.dirname(__file__), output_filename)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    
    # 推論モードに設定
    model.eval()
    
    # ダミー入力の作成
    dummy_input = torch.randn(1, 86, BINS)
    
    # ONNXへエクスポート
    torch.onnx.export(
        model,                  # モデルインスタンス
        dummy_input,            # モデルの入力
        out_path,                  # 出力ファイルパス
        export_params=True,     # モデルのトレーニング済み重みも保存
        opset_version=17,       # ONNXのバージョン
        do_constant_folding=True,  # 定数畳み込み最適化
        input_names=['input'],  # 入力の名前
        output_names=['output'],  # 出力の名前
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'seq_length'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX format at: {out_path}")
    
    # オプション: ONNXモデルの検証
    import onnx
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model checked and verified.")

if __name__ == "__main__":
    main()
    # toOnnx("cqt_transformer_final.pt", "model.onnx")