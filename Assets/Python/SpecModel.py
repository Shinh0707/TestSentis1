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
SAMPLE_RATE = 44100
N_FFT = 1024
HOP_LENGTH = 256
STACK_SIZE = 8
HIDDEN_SIZE = 256
NUM_LAYERS = 3
NUM_HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
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


class ComplexSpectrogramDataset(Dataset):
    """振幅と位相の両方を扱うスペクトログラムデータセット"""

    def __init__(self, audio_files, n_fft=1024, hop_length=256, stack_size=8):
        self.audio_files = audio_files
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stack_size = stack_size

        # 各ファイルの有効なフレーム数を計算
        self.file_frames = []
        self.cumulative_frames = [0]

        for file in tqdm(audio_files, desc="音声ファイル分析中"):
            # 音声読み込み
            y, _ = librosa.load(file, sr=SAMPLE_RATE, mono=True)

            # スペクトログラムに変換
            spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

            # 有効なフレーム数計算
            n_frames = max(0, spec.shape[1] - stack_size)

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

        # 複素スペクトログラムに変換
        complex_spec = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)

        # 振幅スペクトログラム（対数スケール）
        mag_spec = np.log1p(np.abs(complex_spec))

        # 位相スペクトログラム
        # 位相は-π〜πの範囲なので、[-1, 1]に正規化
        phase_spec = np.angle(complex_spec) / np.pi

        # 入力シーケンス（stack_size個の連続フレーム）
        input_mag_frames = []
        input_phase_frames = []

        for i in range(self.stack_size):
            mag_frame = mag_spec[:, frame_idx + i]
            phase_frame = phase_spec[:, frame_idx + i]

            input_mag_frames.append(mag_frame)
            input_phase_frames.append(phase_frame)

        # ターゲット（次のフレーム）
        target_mag_frame = mag_spec[:, frame_idx + self.stack_size]
        target_phase_frame = phase_spec[:, frame_idx + self.stack_size]

        # テンソルに変換
        input_mag_tensor = torch.FloatTensor(np.array(input_mag_frames))
        input_phase_tensor = torch.FloatTensor(np.array(input_phase_frames))

        target_mag_tensor = torch.FloatTensor(target_mag_frame)
        target_phase_tensor = torch.FloatTensor(target_phase_frame)

        # 入力は振幅と位相を連結
        input_tensor = torch.cat([input_mag_tensor, input_phase_tensor], dim=-1)
        target_tensor = torch.cat([target_mag_tensor, target_phase_tensor], dim=-1)

        return input_tensor, target_tensor


class ComplexSpectrogramTransformer(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, num_heads=4, dropout=0.1
    ):
        super().__init__()
        self.input_size = input_size  # n_fft//2+1
        self.hidden_size = hidden_size

        # 入力サイズが2倍（振幅+位相）
        self.embedding = nn.Linear(input_size * 2, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # 出力も2倍（振幅+位相）
        self.output_layer = nn.Linear(hidden_size, input_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: [batch_size, seq_len, input_size*2]
        embedded = self.embedding(src)
        embedded = self.pos_encoder(embedded)

        transformer_output = self.transformer_encoder(embedded, src_mask)
        last_state = transformer_output[:, -1, :]

        output = self.output_layer(last_state)

        return output


# 位相についての循環損失関数
class PhaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 位相差の計算（位相は-1〜1に正規化済み）
        # 元の単位に戻す（-π〜π）
        phase_diff = (pred - target) * np.pi

        # 循環差分にする（-π〜πの範囲にする）
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

        # 二乗平均誤差
        return torch.mean(phase_diff**2)


def train_model():
    # トレーニングデータのパス
    audio_files = glob.glob("Train/*.mp3")
    if not audio_files:
        print("音声ファイルが見つかりません")
        return

    print(f"検出された音声ファイル: {len(audio_files)}個")

    # データセットとデータローダーの作成
    dataset = ComplexSpectrogramDataset(audio_files, N_FFT, HOP_LENGTH, STACK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # スペクトログラムの特徴量サイズ
    input_size = N_FFT // 2 + 1

    # モデルの初期化
    model = ComplexSpectrogramTransformer(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # 損失関数（振幅用と位相用）
    mag_criterion = nn.MSELoss()
    phase_criterion = PhaseLoss()

    # 最適化アルゴリズム
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

            predicted_frames = model(input_frames)

            # 振幅と位相を分ける
            pred_mag = predicted_frames[:, :input_size]
            pred_phase = predicted_frames[:, input_size:]

            target_mag = target_frames[:, :input_size]
            target_phase = target_frames[:, input_size:]

            # それぞれの損失計算
            mag_loss = mag_criterion(pred_mag, target_mag)
            phase_loss = phase_criterion(pred_phase, target_phase)

            # 合計損失（振幅の損失を優先）
            loss = mag_loss + 0.5 * phase_loss

            # 勾配計算とパラメータ更新
            loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            # 損失の記録
            epoch_loss += loss.item()
            batch_count += 1

            # プログレスバーの更新
            progress_bar.set_postfix(
                loss=loss.item(), mag_loss=mag_loss.item(), phase_loss=phase_loss.item()
            )

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
                f"complex_spectro_transformer_epoch_{epoch+1}.pth",
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
            "input_size": input_size,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
            "stack_size": STACK_SIZE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
        },
        "complex_spectro_transformer_final.pth",
    )

    return model, train_losses


def test_realtime_simulation(model, test_file, output_file, duration_seconds=10):
    """振幅と位相の両方を予測するリアルタイム処理シミュレーション"""
    model.eval()

    # テスト音声の読み込み
    y, _ = librosa.load(test_file, sr=SAMPLE_RATE, mono=True)

    # スペクトログラムに変換
    complex_spec = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # 振幅（対数スケール）と位相を抽出
    mag_spec = np.log1p(np.abs(complex_spec))
    phase_spec = np.angle(complex_spec) / np.pi  # -1〜1に正規化

    # 最小必要フレーム数をチェック
    if mag_spec.shape[1] < STACK_SIZE + 1:
        print("テスト音声が短すぎます")
        return

    # 初期バッファの準備
    buffer_mag = []
    buffer_phase = []

    for i in range(STACK_SIZE):
        buffer_mag.append(mag_spec[:, i])
        buffer_phase.append(phase_spec[:, i])

    # 入力の特徴量サイズ
    input_size = mag_spec.shape[0]

    # 出力スペクトログラム用のバッファ
    output_mag = np.copy(mag_spec[:, :STACK_SIZE])
    output_phase = np.copy(phase_spec[:, :STACK_SIZE])

    # シミュレーション時間の測定用
    processing_times = []

    # 処理するフレーム数
    max_frames = min(
        int(duration_seconds * SAMPLE_RATE / HOP_LENGTH), mag_spec.shape[1] - STACK_SIZE
    )

    print(f"リアルタイム処理シミュレーション開始：{max_frames}フレーム")

    for i in tqdm(range(max_frames)):
        # 現在のバッファを連結してテンソルに変換
        buffer_combined = []
        for j in range(STACK_SIZE):
            # 各フレームで振幅と位相を連結
            combined = np.concatenate([buffer_mag[j], buffer_phase[j]])
            buffer_combined.append(combined)

        input_tensor = (
            torch.FloatTensor(np.array(buffer_combined)).unsqueeze(0).to(DEVICE)
        )

        # 処理開始時間
        start_time = time.time()

        # モデルによる予測
        with torch.no_grad():
            predicted_combined = model(input_tensor)
            predicted_combined = predicted_combined.cpu().numpy().squeeze()

        # 予測結果を振幅と位相に分離
        predicted_mag = predicted_combined[:input_size]
        predicted_phase = predicted_combined[input_size:]

        # 処理終了時間と所要時間計算
        end_time = time.time()
        process_time = end_time - start_time
        processing_times.append(process_time)

        # 予測フレームを出力バッファに追加
        output_mag = np.column_stack((output_mag, predicted_mag))
        output_phase = np.column_stack((output_phase, predicted_phase))

        # 次のフレームのために入力バッファを更新
        buffer_mag.pop(0)
        buffer_mag.append(predicted_mag)

        buffer_phase.pop(0)
        buffer_phase.append(predicted_phase)

    # 処理時間統計
    avg_time = np.mean(processing_times)
    max_time = np.max(processing_times)
    min_time = np.min(processing_times)

    print(f"平均処理時間: {avg_time*1000:.2f}ms/フレーム")
    print(f"最大処理時間: {max_time*1000:.2f}ms/フレーム")
    print(f"最小処理時間: {min_time*1000:.2f}ms/フレーム")

    # リアルタイム処理可能かどうかの判断
    frame_duration = HOP_LENGTH / SAMPLE_RATE  # 1フレームの時間（秒）
    if avg_time > frame_duration:
        print(
            f"警告: 平均処理時間({avg_time*1000:.2f}ms)がフレーム長({frame_duration*1000:.2f}ms)を超えています。リアルタイム処理が難しい可能性があります。"
        )
    else:
        margin = (frame_duration - avg_time) / frame_duration * 100
        print(f"リアルタイム処理可能: {margin:.1f}%の余裕があります")

    # スペクトログラムから波形に変換
    # 対数振幅を元に戻す
    output_mag = np.expm1(output_mag)

    # 位相を-π〜πの範囲に戻す
    output_phase = output_phase * np.pi

    # 振幅と位相から複素数スペクトログラムを再構成
    output_complex_spec = output_mag * np.exp(1j * output_phase)

    # 逆STFTで波形に変換
    output_audio = librosa.istft(
        output_complex_spec, hop_length=HOP_LENGTH, win_length=N_FFT
    )

    # 音量の正規化
    output_audio = output_audio / np.max(np.abs(output_audio))

    # ファイル保存
    sf.write(output_file, output_audio, SAMPLE_RATE)

    print(f"生成音声を保存しました: {output_file}")

    return output_audio, processing_times


if __name__ == "__main__":
    # モデルのトレーニング
    model, losses = train_model()

    # テスト用ファイルがあればテスト実行
    test_files = glob.glob("Test/*.mp3")
    if test_files:
        print(f"リアルタイムシミュレーションテスト: {test_files[0]}")
        test_realtime_simulation(
            model,
            test_file=test_files[0],
            output_file="complex_spectro_generated_audio.wav",
            duration_seconds=30,
        )
    else:
        print("テスト用音声ファイルが見つかりません")
