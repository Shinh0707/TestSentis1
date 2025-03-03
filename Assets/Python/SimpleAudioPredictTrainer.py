import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
import time
import math

plt.rcParams["font.family"] = "MS Gothic"


# パラメータ設定
SAMPLE_RATE = 44100  # サンプルレート
FRAME_SIZE = 1024  # 1フレームのサンプル数
STACK_SIZE = 44100 // 1024
HIDDEN_SIZE = 512
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """
    トランスフォーマーの位置エンコーディング
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class RawAudioDataset(Dataset):
    """
    生の音声波形データを扱うデータセット
    """

    def __init__(self, audio_files, frame_size=1024, stack_size=8):
        self.audio_files = audio_files
        self.frame_size = frame_size
        self.stack_size = stack_size

        # 各ファイルの有効なフレーム数を計算
        self.file_frames = []
        self.cumulative_frames = [0]

        for file in tqdm(audio_files, desc="音声ファイル分析中"):
            # 音声読み込み
            y, _ = librosa.load(file, sr=SAMPLE_RATE, mono=True)

            # フレーム数を計算（オーバーラップなしでフレーム分割）
            n_frames = (len(y) // frame_size) - stack_size
            if n_frames > 0:
                self.file_frames.append((file, n_frames))
                self.cumulative_frames.append(self.cumulative_frames[-1] + n_frames)

        print(f"全ファイルの有効フレーム数: {self.cumulative_frames[-1]}")

    def __len__(self):
        return self.cumulative_frames[-1]

    def __getitem__(self, idx):
        # どのファイルの何番目のフレームかを特定
        file_idx = 0
        while idx >= self.cumulative_frames[file_idx + 1]:
            file_idx += 1

        # ファイル内でのフレーム位置
        frame_idx = idx - self.cumulative_frames[file_idx]
        filename, _ = self.file_frames[file_idx]

        # 音声データ読み込み
        y, _ = librosa.load(filename, sr=SAMPLE_RATE, mono=True)

        # フレーム開始位置
        start_pos = frame_idx * self.frame_size

        # 入力シーケンス（stack_size個の連続フレーム）
        input_frames = []
        for i in range(self.stack_size):
            frame_start = start_pos + (i * self.frame_size)
            frame_end = frame_start + self.frame_size
            frame = y[frame_start:frame_end]
            input_frames.append(frame)

        # ターゲット（次のフレーム）
        target_start = start_pos + (self.stack_size * self.frame_size)
        target_end = target_start + self.frame_size
        target_frame = y[target_start:target_end]

        # テンソルに変換
        input_tensor = torch.FloatTensor(np.array(input_frames))
        target_tensor = torch.FloatTensor(target_frame)

        return input_tensor, target_tensor


def create_padding_mask(src, pad_idx=0):
    """
    パディングマスクの作成
    """
    # src: [batch_size, seq_len, features]
    mask = (src[:, :, 0] == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


class AudioTransformer(nn.Module):
    def __init__(
        self, frame_size, hidden_size=128, num_layers=2, num_heads=4, dropout=0.1
    ):
        super().__init__()
        self.frame_size = frame_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(frame_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        self.output_layer = nn.Linear(hidden_size, frame_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        embedded = self.embedding(src)
        embedded = self.pos_encoder(embedded)

        transformer_output = self.transformer_encoder(embedded, src_mask)
        last_state = transformer_output[:, -1, :]

        output = self.output_layer(last_state)

        return output


def train_model():
    # トレーニングデータのパス
    audio_files = glob.glob("Train/*.mp3")
    if not audio_files:
        print("音声ファイルが見つかりません")
        return

    print(f"検出された音声ファイル: {len(audio_files)}個")

    # データセットとデータローダーの作成
    dataset = RawAudioDataset(audio_files, FRAME_SIZE, STACK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # モデルの初期化
    model = AudioTransformer(
        frame_size=FRAME_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # 損失関数と最適化アルゴリズム
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学習率スケジューラー
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 損失の履歴
    train_losses = []

    # トレーニングループ
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        # エポック開始時間
        start_time = time.time()

        for batch_idx, (input_frames, target_frames) in enumerate(progress_bar):
            input_frames = input_frames.to(DEVICE)
            target_frames = target_frames.to(DEVICE)

            # モデルによる予測
            optimizer.zero_grad()

            predicted_frames_raw = model(input_frames)

            loss = criterion(predicted_frames_raw, target_frames)

            # 勾配計算とパラメータ更新
            loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            # 損失の記録
            epoch_loss += loss.item()
            batch_count += 1

            # プログレスバーの更新
            progress_bar.set_postfix(loss=loss.item())

        # エポックの平均損失
        avg_epoch_loss = epoch_loss / batch_count
        train_losses.append(avg_epoch_loss)

        # エポック終了時間と処理時間
        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s"
        )

        # 学習率の調整
        scheduler.step(avg_epoch_loss)

        # モデルの保存（5エポックごと）
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                },
                f"audio_transformer_epoch_{epoch+1}.pth",
            )

    # 学習曲線のプロット
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.close()

    # 最終モデルの保存
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "frame_size": FRAME_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
            "stack_size": STACK_SIZE,
        },
        "audio_transformer_final.pth",
    )

    return model, train_losses


def test_realtime_simulation(model, test_file, output_file, duration_seconds=10):
    """
    デノイジング機能を含むリアルタイム処理のシミュレーション
    """
    model.eval()

    # テスト音声の読み込み
    y, _ = librosa.load(test_file, sr=SAMPLE_RATE, mono=True)

    # 入力バッファとして使用する初期フレーム
    if len(y) < FRAME_SIZE * STACK_SIZE:
        print("テスト音声が短すぎます")
        return

    # 初期バッファの準備
    buffer = []
    for i in range(STACK_SIZE):
        start = i * FRAME_SIZE
        end = start + FRAME_SIZE
        buffer.append(y[start:end])

    # 出力音声用のバッファ
    output_audio = []
    # output_audio.extend(buffer[0])

    # シミュレーション時間の測定用
    processing_times = []

    # フレーム数
    n_frames = int(duration_seconds * SAMPLE_RATE / FRAME_SIZE)

    print(f"リアルタイム処理シミュレーション開始：{n_frames}フレーム")

    for i in tqdm(range(n_frames)):
        # 現在のバッファをテンソルに変換
        input_tensor = torch.FloatTensor(np.array(buffer)).unsqueeze(0).to(DEVICE)
        # 処理開始時間
        start_time = time.time()

        with torch.no_grad():
            predicted_frame = model.forward(input_tensor)
            predicted_frame = predicted_frame.cpu().numpy().squeeze()

        # 処理終了時間と所要時間計算
        end_time = time.time()
        process_time = end_time - start_time
        processing_times.append(process_time)

        # 予測フレームを出力バッファに追加
        output_audio.extend(predicted_frame)

        # 次のフレームのために入力バッファを更新
        buffer.pop(0)
        buffer.append(predicted_frame)

    # 処理時間統計
    avg_time = np.mean(processing_times)
    max_time = np.max(processing_times)
    min_time = np.min(processing_times)

    print(f"平均処理時間: {avg_time*1000:.2f}ms/フレーム")
    print(f"最大処理時間: {max_time*1000:.2f}ms/フレーム")
    print(f"最小処理時間: {min_time*1000:.2f}ms/フレーム")

    # リアルタイム処理可能かどうかの判断
    frame_duration = FRAME_SIZE / SAMPLE_RATE  # 1フレームの時間（秒）
    if avg_time > frame_duration:
        print(
            f"警告: 平均処理時間({avg_time*1000:.2f}ms)がフレーム長({frame_duration*1000:.2f}ms)を超えています。リアルタイム処理が難しい可能性があります。"
        )
    else:
        margin = (frame_duration - avg_time) / frame_duration * 100
        print(f"リアルタイム処理可能: {margin:.1f}%の余裕があります")

    # 出力音声を保存
    output_audio = np.array(output_audio)

    # 音量の正規化
    output_audio = output_audio / np.max(np.abs(output_audio))
    # ファイル保存
    sf.write(output_file, output_audio, SAMPLE_RATE)

    print(f"生成音声を保存しました: {output_file}")

    """
    # 入力と出力の波形プロット
    plt.figure(figsize=(12, 12))

    # 入力音声
    plt.subplot(2, 1, 1)
    plt.plot(y[: len(output_audio)])
    plt.title("入力音声波形")
    plt.xlabel("サンプル")
    plt.ylabel("振幅")

    # 出力音声
    plt.subplot(2, 1, 2)
    plt.plot(output_audio)
    plt.title("生成音声波形")
    plt.xlabel("サンプル")
    plt.ylabel("振幅")

    plt.tight_layout()
    plt.savefig("processed_audio_waveforms_comparison.png")

    # スペクトログラムのプロット
    plt.figure(figsize=(12, 12))

    # 入力音声のスペクトログラム
    plt.subplot(2, 1, 1)
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y[: len(output_audio)])), ref=np.max
    )
    librosa.display.specshow(D, sr=SAMPLE_RATE, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("入力音声スペクトログラム")

    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(output_audio)), ref=np.max)
    librosa.display.specshow(D, sr=SAMPLE_RATE, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("生成音声スペクトログラム")

    plt.tight_layout()
    plt.savefig("processed_audio_spectrograms_comparison.png")
    """

    return output_audio, processing_times


if __name__ == "__main__":
    # モデルのトレーニング
    model, losses = train_model()
    """
    model = AudioTransformer(
        frame_size=FRAME_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load("audio_transformer_final.pth")["model_state_dict"])
    """
    # テスト用ファイルがあればテスト実行
    test_files = glob.glob("Test/*.mp3")
    if test_files:
        print(f"リアルタイムシミュレーションテスト: {test_files[0]}")
        test_realtime_simulation(
            model,
            test_file=test_files[0],
            output_file="realtime_generated_audio.wav",
            duration_seconds=30,
        )
    else:
        print("テスト用音声ファイルが見つかりません")
