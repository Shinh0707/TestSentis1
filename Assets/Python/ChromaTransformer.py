import torch
import torch.nn as nn
from torch import Tensor, softmax
from torch.nn import Module, Sequential, Linear, ReLU, Dropout
import math


def attn_map(q: Tensor, k: Tensor):
    return q @ k.transpose(-2, -1) / q.size(-1)


def triu_flatten(x: Tensor, diagonal=1):
    """
    バッチ処理対応の上三角部分flattenライブラリ（ベクトル化実装）

    B×D×D の入力テンソルから、バッチごとに上三角部分をflattenして
    B×(D*(D-1))/2 の形状のテンソルを返す。

    Args:
        x: 入力テンソル [batch_size, D, D] (torch.Tensor)
        diagonal: 対角成分のオフセット (default=1: 対角成分を除外)

    Returns:
        上三角部分をflattenしたテンソル [batch_size, (D*(D-1))/2]（diagonal=1の場合）
        または [batch_size, (D*(D+1))/2]（diagonal=0の場合）
    """
    batch_size, d1, d2 = x.shape
    # 上三角マスクを作成
    mask = torch.triu(torch.ones(d1, d2, device=x.device), diagonal=diagonal).bool()
    # 行列全体をviewしてから上三角部分を抽出
    # reshapeで最初の次元をバッチとして保持し、残りの次元を結合して上三角部分を抽出
    return x.reshape(batch_size, -1)[:, mask.reshape(-1)]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout=0.1):
        """
        Positional Encodingモジュール

        Args:
            d_model: モデルの埋め込み次元 (int)
            max_seq_length: 最大シーケンス長 (int, default=5000)
            dropout: ドロップアウト率 (float, default=0.1)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ポジショナルエンコーディング行列の計算
        pe = torch.zeros(max_seq_length, d_model, requires_grad=False)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数インデックスにはsinを適用
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数インデックスにはcosを適用（d_modelが奇数の場合は最後の要素を考慮）
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model + 1) // 2])
        self.pe = pe.unsqueeze(0)

    def forward(self, x: Tensor):
        """
        Args:
            x: 入力テンソル [batch_size, seq_length, d_model]

        Returns:
            ポジショナルエンコーディングを加えたテンソル [batch_size, seq_length, d_model]
        """
        # 入力テンソルのシーケンス長に合わせて加算
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout.forward(x)


class Attension(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        return softmax(attn_map(q, k), dim=-1) @ v


class CQTTransformerLayer(Module):
    def __init__(
        self, nbins: int, max_seq_len: int = 1000, dropout: float = 0.1, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.attn = Attension()
        self.pe = PositionalEncoding(nbins, max_seq_len)
        self.ls1 = Sequential(
            Linear(nbins, nbins * 2), ReLU(), Linear(nbins * 2, nbins), Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pe.forward(x)
        x = self.attn.forward(x, x, x)
        x = x + self.ls1.forward(x)
        return x


class CQTTransformer(Module):
    def __init__(
        self,
        nbins: int,
        layers: int = 4,
        max_seq_len: int = 1000,
        dropout: float = 0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.chroma_dim = nbins
        self.max_seq_len = max_seq_len
        self.attn = Attension()
        self.pe = PositionalEncoding(nbins, self.max_seq_len)
        self.layers = Sequential(
            *[
                CQTTransformerLayer(
                    nbins=nbins, max_seq_len=max_seq_len, dropout=dropout
                )
                for _ in range(layers)
            ]
        )
        self.ls2 = Sequential(
            Linear(nbins, self.chroma_dim * 2),
            Dropout(dropout),
            Linear(self.chroma_dim * 2, self.chroma_dim),
            ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers.forward(x)
        x: Tensor = self.ls2.forward(x)
        return x.mean(-2)
