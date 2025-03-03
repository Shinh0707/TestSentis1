import torch
import torch.nn as nn
import numpy as np


class AudioPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, stack_size):
        super(AudioPredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.stack_size = stack_size

        # スタックされた入力とhidden state用のLSTM
        self.lstm = nn.LSTM(
            input_size=input_size * stack_size,  # スタックされたオーディオフレーム
            hidden_size=hidden_size,
            batch_first=True,
        )

        # 出力層（次のオーディオフレームを予測）
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, audio_stack, hidden=None):
        """
        audio_stack: (batch_size, stack_size, input_features)のスタックされたオーディオフレーム
        hidden: (h0, c0) - 前回のLSTMの状態
        """
        batch_size = audio_stack.size(0)

        # スタックされたフレームを1つのフレームとしてLSTMに入力
        # (batch_size, stack_size, features) -> (batch_size, 1, stack_size*features)
        x = audio_stack.reshape(batch_size, 1, -1)

        # LSTMに通す（前回の状態を引き継ぐ）
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)

        # 出力を次のオーディオフレームに変換
        next_frame = self.output_layer(out.squeeze(1))

        return next_frame, hidden


# 使用例
def predict_next_audio_frame(model, audio_buffer, hidden_state=None):
    """
    リアルタイムストリームから次のオーディオフレームを予測

    audio_buffer: 最新のstack_size分のオーディオフレームを含むバッファ
    hidden_state: 前回の隠れ状態とセル状態のタプル
    """
    # バッファをテンソルに変換
    audio_stack = torch.FloatTensor(audio_buffer).unsqueeze(0)  # バッチ次元を追加

    # 予測実行
    with torch.no_grad():
        next_frame, new_hidden = model(audio_stack, hidden_state)

    return next_frame.numpy().squeeze(), new_hidden


# リアルタイム予測のシミュレーション
def audio_stream_prediction_demo():
    # モデル設定
    input_size = 1024
    hidden_size = 1024
    stack_size = 5  # 過去何フレームを考慮するか

    # モデル初期化
    model = AudioPredictionLSTM(input_size, hidden_size, stack_size)
    model.eval()

    # 初期バッファ（過去のオーディオフレーム）
    audio_buffer = np.random.randn(stack_size, input_size).astype(np.float32)

    # 初期状態
    hidden_state = None

    # オーディオストリームシミュレーション
    for i in range(10):  # 10フレーム分シミュレート
        # 次のフレームを予測
        predicted_frame, hidden_state = predict_next_audio_frame(
            model, audio_buffer, hidden_state
        )

        print(f"フレーム {i+1} 予測完了")

        # 実際のオーディオフレームを取得（ここではダミーデータ）
        # 実際のアプリケーションではマイクからの入力など
        real_frame = np.random.randn(input_size).astype(np.float32)

        # バッファを更新（最も古いフレームを捨て、新しいフレームを追加）
        audio_buffer = np.vstack([audio_buffer[1:], real_frame])

    return "予測完了"
