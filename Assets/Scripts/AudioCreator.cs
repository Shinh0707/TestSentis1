using UnityEngine;
using System;
using Ext;

public class AudioCreator : MonoBehaviour
{
    [SerializeField] private CQTTransformer generator;
    
    // マイク関連の設定
    [SerializeField] private string microphoneDeviceName = null; // null で既定のマイクを使用
    [SerializeField] private int microphoneSampleRate = 44100;
    [SerializeField] private int stacks = 86;
    [SerializeField] private int bufferLengthSec = 2;  // マイクバッファの長さ(秒)
    
    private AudioClip microphoneClip;
    private int lastSamplePosition = 0;
    private float[] sampleBuffer;
    private int sampleBufferSize => generator.bufferLength;

    void Start()
    {
        if (generator == null)
        {
            generator = GetComponent<CQTTransformer>();
            if (generator == null)
            {
                generator = gameObject.AddComponent<CQTTransformer>();
            }
        }

        generator.Initialize(stacks);
        sampleBuffer = new float[sampleBufferSize];
        
        StartMicrophoneCapture();
    }

    void StartMicrophoneCapture()
    {
        // マイクが使用可能か確認
        if (Microphone.devices.Length == 0)
        {
            Debug.LogError("マイクが見つかりません");
            return;
        }

        // デバイス名が指定されていなければ最初のデバイスを使用
        if (string.IsNullOrEmpty(microphoneDeviceName))
        {
            microphoneDeviceName = Microphone.devices[0];
            Debug.Log($"使用するマイク: {microphoneDeviceName}");
        }

        // マイクからの録音を開始
        microphoneClip = Microphone.Start(microphoneDeviceName, true, bufferLengthSec, microphoneSampleRate);

        // マイクの準備ができるまで待機
        while (!(Microphone.GetPosition(microphoneDeviceName) > 0)) { }

        lastSamplePosition = 0;
        Debug.Log("マイク入力の準備完了");
    }

    void Update()
    {
        if (microphoneClip == null) return;

        // 現在のマイク位置を取得
        int currentPosition = Microphone.GetPosition(microphoneDeviceName);
        
        // 新しいサンプルがあるか確認
        if (currentPosition != lastSamplePosition)
        {
            // ストリーミングデータを処理
            ProcessAudioStream(currentPosition);
        }
    }

    private void ProcessAudioStream(int currentPosition)
    {
        int totalSamples = microphoneClip.samples;
        
        // バッファが一周して戻ってきたか確認
        bool wrapped = currentPosition < lastSamplePosition;
        
        // 処理すべきサンプル数を計算
        int samplesToProcess;
        if (wrapped)
        {
            samplesToProcess = (totalSamples - lastSamplePosition) + currentPosition;
        }
        else
        {
            samplesToProcess = currentPosition - lastSamplePosition;
        }
        
        // バッファサイズごとに処理
        while (samplesToProcess > 0)
        {
            int samplesToRead = Math.Min(samplesToProcess, sampleBufferSize);
            
            if (wrapped && lastSamplePosition + samplesToRead > totalSamples)
            {
                // バッファの終端から先頭へのラップアラウンド処理
                int samplesUntilEnd = totalSamples - lastSamplePosition;
                int samplesFromStart = samplesToRead - samplesUntilEnd;
                
                // 終端部分を読み込み
                microphoneClip.GetData(sampleBuffer, lastSamplePosition);
                
                // 先頭部分を読み込んで追加
                if (samplesFromStart > 0)
                {
                    float[] tempBuffer = new float[samplesFromStart];
                    microphoneClip.GetData(tempBuffer, 0);
                    Array.Copy(tempBuffer, 0, sampleBuffer, samplesUntilEnd, samplesFromStart);
                }
                
                // 次の読み込み位置を更新
                lastSamplePosition = samplesFromStart;
            }
            else
            {
                // 通常の連続読み込み
                microphoneClip.GetData(sampleBuffer, lastSamplePosition);
                
                // 次の読み込み位置を更新
                lastSamplePosition = (lastSamplePosition + samplesToRead) % totalSamples;
            }
            
            // CQT処理に送信
            generator.SetInput(sampleBuffer);
            
            // 残りのサンプル数を更新
            samplesToProcess -= samplesToRead;
        }
    }
}
