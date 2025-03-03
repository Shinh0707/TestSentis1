using Unity.Sentis;
using System;
using System.Numerics;


namespace Ext{
public class ConstantQTransform
{
    private readonly double _minFrequency;    // 最小周波数 (Hz)
    private readonly double _maxFrequency;    // 最大周波数 (Hz)
    private readonly int _binsPerOctave;      // オクターブあたりのビン数
    private int _maxKernelLength;
    private readonly double _q;               // Q値 (周波数分解能)
    private readonly int _sampleRate;         // サンプリングレート (Hz)
    private readonly int _totalBins;          // 総ビン数
    private readonly int _hopLength;          // ホップ長
    private readonly Complex[][] _kernels;    // 各ビンのカーネル

    /// <summary>
    /// Constant Q Transform の初期化
    /// </summary>
    public ConstantQTransform(double minFrequency=110, double maxFrequency=14080, int binsPerOctave=12, 
        int sampleRate=44100, int hopLength=512, double q = 1.0)
    {
        _minFrequency = minFrequency;
        _maxFrequency = maxFrequency;
        _binsPerOctave = binsPerOctave;
        _hopLength = hopLength;
        _maxKernelLength = 0;
        _sampleRate = sampleRate;
        _q = q;

        // 総ビン数を計算
        _totalBins = (int)Math.Ceiling(binsPerOctave * Math.Log(_maxFrequency / _minFrequency, 2));
        
        // カーネルを初期化
        _kernels = new Complex[_totalBins][];
        
        InitializeKernels();
    }

    /// <summary>
    /// CQT変換用のカーネルを初期化
    /// </summary>
    private void InitializeKernels()
    {
        _maxKernelLength = 0;
        for (int k = 0; k < _totalBins; k++)
        {
            // k番目のビンの中心周波数を計算
            double centerFreq = _minFrequency * Math.Pow(2, (double)k / _binsPerOctave);
            
            // 窓関数の長さを計算（周波数に反比例）
            int windowLength = (int)Math.Ceiling(_q * _sampleRate / centerFreq);
            
            // 窓長が偶数の場合は奇数に調整
            if (windowLength % 2 == 0) windowLength++;
            
            _kernels[k] = new Complex[windowLength];
            
            // カーネルの生成（複素正弦波 × 窓関数）
            double windowCenter = (windowLength - 1) / 2.0;
            for (int n = 0; n < windowLength; n++)
            {
                // 位相
                double phase = 2.0 * Math.PI * centerFreq * (n - windowCenter) / _sampleRate;
                
                // ハミング窓
                double window = 0.54 - 0.46 * Math.Cos(2.0 * Math.PI * n / (windowLength - 1));
                
                // 窓関数適用済みの複素正弦波
                _kernels[k][n] = new Complex(
                    Math.Cos(phase) * window,
                    Math.Sin(phase) * window
                );
            }
            
            // カーネルの正規化
            double normFactor = 0;
            foreach (var value in _kernels[k])
            {
                normFactor += value.Magnitude * value.Magnitude;
            }
            normFactor = Math.Sqrt(normFactor);
            
            for (int n = 0; n < windowLength; n++)
            {
                _kernels[k][n] /= normFactor;
            }
            _maxKernelLength = Math.Max(_maxKernelLength, _kernels[k].Length);
        }
    }

    public Complex[,] CreateResultBuffer(int bufferLength){
        int numFrames = 1 + (bufferLength - _maxKernelLength) / _hopLength;
        numFrames = Math.Max(1, numFrames);
        return new Complex[numFrames,_totalBins];
    }
    public float[] CreateOutputBuffer(){
        return new float[_totalBins];
    }

    /// <summary>
    /// オーディオデータにCQT変換を適用し、ホップ長を考慮してスペクトログラムを生成
    /// </summary>
    public void Transform(float[] audioData, Complex[,] cqtSpectrogram)
    {
        // フレーム数の計算
        int numFrames = 1 + (audioData.Length - _maxKernelLength) / _hopLength;

        int t, frameStart, k, n, kernelLength, audioIndex;
        for (t = 0; t < numFrames; t++)
        {
            // 現在のフレームの開始位置
            frameStart = t * _hopLength;
            
            for (k = 0; k < _totalBins; k++)
            {
                cqtSpectrogram[t,k] = 0;
                kernelLength = _kernels[k].Length;
                
                for (n = 0; n < kernelLength; n++)
                {
                    audioIndex = frameStart + n;
                    
                    // オーディオデータの範囲内かチェック
                    if (audioIndex < audioData.Length)
                    {
                        cqtSpectrogram[t,k] += (double)audioData[audioIndex] * Complex.Conjugate(_kernels[k][n]);
                    }
                }
            }
        }
    }

    /// <summary>
    /// 単一フレームのCQT変換を計算
    /// </summary>
    public Complex[] TransformSingleFrame(double[] frame)
    {
        var result = new Complex[_totalBins];
        
        for (int k = 0; k < _totalBins; k++)
        {
            Complex sum = 0;
            int kernelLength = _kernels[k].Length;
            
            // 入力フレームの長さがカーネル長より短い場合に対処
            int processLength = Math.Min(frame.Length, kernelLength);
            
            for (int n = 0; n < processLength; n++)
            {
                sum += frame[n] * Complex.Conjugate(_kernels[k][n]);
            }
            
            result[k] = sum;
        }
        
        return result;
    }

    /// <summary>
    /// 各ビンの中心周波数（Hz）を取得
    /// </summary>
    public double[] GetCenterFrequencies()
    {
        var frequencies = new double[_totalBins];
        for (int k = 0; k < _totalBins; k++)
        {
            frequencies[k] = _minFrequency * Math.Pow(2, (double)k / _binsPerOctave);
        }
        return frequencies;
    }
}
public static class Signals{
    public static float[] CreateFloatRawData(Complex[,] data){
        return new float[data.GetLength(0)*data.GetLength(1)];
    }

    public static double[,] CreateDoubleRawData(Complex[,] data){
        return new double[data.GetLength(0),data.GetLength(1)];
    }

    public static void Flatten(double[,] data, float[] dst){
        int j;
        int imax = data.GetLength(0);
        int jmax = data.GetLength(1);
        for(int i = 0;i < imax; i++){
            for(j = 0;j < jmax; j++){
                dst[i*jmax+j] = (float)data[i,j];
            }
        }
    }

    public static void Abs(Complex[,] signals, double[,] res){
        int j;
        int imax = signals.GetLength(0);
        int jmax = signals.GetLength(1);
        for(int i = 0;i < imax; i++){
            for(j = 0;j < jmax; j++){
                res[i,j] = Complex.Abs(signals[i,j]);
            }
        }
    }

    public static void StdNorm(double[,] signals){
        int i,j;
        int imax = signals.GetLength(0);
        int jmax = signals.GetLength(1);
        double[,] prms = new double[imax,2];
        for (i = 0; i < imax; i++){
            prms[i,0] = 0;
            for (j = 0; j < jmax; j++){
                prms[i,0] += signals[i,j];
            }
            prms[i,0] /= imax;
        }
        for (i = 0; i < imax; i++){
            prms[i,1] = 0;
            for (j = 0; j < jmax; j++){
                prms[i,1] = signals[i,j] - prms[i,0];
            }
            prms[i,1] /= imax;
        }
        for (i = 0; i < imax; i++){
            for (j = 0; j < jmax; j++){
                signals[i,j] = (signals[i,j] - prms[i,0])/(prms[i,1] + 0.00001);
            }
        }
    }
    public static void StdNorm(Complex[,] signals, double[,] midRes, float[] dst){
        int i,j;
        int imax = signals.GetLength(0);
        int jmax = signals.GetLength(1);
        for(i = 0;i < imax; i++){
            for(j = 0;j < jmax; j++){
                midRes[i,j] = Complex.Abs(signals[i,j]);
            }
        }
        double[,] prms = new double[imax,2];
        for (i = 0; i < imax; i++){
            prms[i,0] = 0;
            for (j = 0; j < jmax; j++){
                prms[i,0] += midRes[i,j];
            }
            prms[i,0] /= imax;
        }
        for (i = 0; i < imax; i++){
            prms[i,1] = 0;
            for (j = 0; j < jmax; j++){
                prms[i,1] = midRes[i,j] - prms[i,0];
            }
            prms[i,1] /= imax;
        }
        for (i = 0; i < imax; i++){
            for (j = 0; j < jmax; j++){
                dst[i*jmax+j] = (float)((midRes[i,j] - prms[i,0])/(prms[i,1] + 0.00001));
            }
        }
    }
}
}
