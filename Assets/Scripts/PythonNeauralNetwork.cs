using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.Sentis;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace AI.NeuralNetwork
{
    /// <summary>
    /// Pythonで学習されるニューラルネットワークモデルとの連携を管理する抽象クラス
    /// </summary>
    public abstract class PythonNeuralNetworkBase : MonoBehaviour
    {
        [Header("基本設定")]
        [SerializeField]
        protected string sharedFolderPath = "SharedFolder";

        [SerializeField]
        protected string pythonPath = "python";

        [SerializeField]
        protected string scriptPath = "model_server.py";

        [SerializeField]
        protected string modelModulePath = "";

        [Header("モデル設定")]
        [SerializeField]
        protected TextAsset initialConfig;

        [SerializeField]
        protected int checkIntervalSeconds = 5;

        protected string modelStatusPath;
        protected string modelOnnxPath;
        protected string csvPath;
        protected string configPath;
        protected string writeLockPath;

        protected Model model;
        protected Worker engine;
        protected bool isModelLoaded = false;
        protected bool isWatcherRunning = false;
        protected CancellationTokenSource watcherCancellationToken;

        protected Queue<InputData> inputDataQueue = new Queue<InputData>();
        protected InputData lastPrediction = null;

        protected virtual void Awake()
        {
            // パスの設定
            sharedFolderPath = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", sharedFolderPath)
            );
            modelStatusPath = Path.Combine(sharedFolderPath, "model_status.json");
            modelOnnxPath = Path.Combine(sharedFolderPath, "model.onnx");
            csvPath = Path.Combine(sharedFolderPath, "InputHistory.csv");
            configPath = Path.Combine(sharedFolderPath, "Config.json");
            writeLockPath = Path.Combine(sharedFolderPath, "write.lock");

            // 共有フォルダの作成
            if (!Directory.Exists(sharedFolderPath))
            {
                Directory.CreateDirectory(sharedFolderPath);
            }

            // 初期設定ファイルの作成
            if (initialConfig != null && !File.Exists(configPath))
            {
                File.WriteAllText(configPath, initialConfig.text);
            }
        }

        protected virtual void OnEnable()
        {
            // モデルウォッチャーを開始
            StartModelWatcher();
        }

        protected virtual void OnDisable()
        {
            // モデルウォッチャーを停止
            StopModelWatcher();

            // モデルのアンロード
            UnloadModel();
        }

        protected virtual void OnDestroy()
        {
            StopModelWatcher();
            UnloadModel();
        }

        /// <summary>
        /// モデルの状態を監視するウォッチャーを開始
        /// </summary>
        protected virtual void StartModelWatcher()
        {
            if (isWatcherRunning)
                return;

            watcherCancellationToken = new CancellationTokenSource();
            isWatcherRunning = true;

            Task.Run(
                async () =>
                {
                    while (!watcherCancellationToken.IsCancellationRequested)
                    {
                        try
                        {
                            ModelStatus status = GetModelStatus();
                            if (status != null && status.status == "ready_to_load")
                            {
                                // モデルが更新され、読み込み準備ができている
                                UnityMainThreadDispatcher.Instance.Enqueue(() =>
                                {
                                    LoadModel();
                                    UpdateModelStatus("loaded");
                                });
                            }

                            // 一定間隔待機
                            await Task.Delay(
                                checkIntervalSeconds * 1000,
                                watcherCancellationToken.Token
                            );
                        }
                        catch (Exception ex)
                        {
                            Debug.LogError($"モデルウォッチャーエラー: {ex.Message}");
                            await Task.Delay(
                                checkIntervalSeconds * 1000,
                                watcherCancellationToken.Token
                            );
                        }
                    }
                },
                watcherCancellationToken.Token
            );
        }

        /// <summary>
        /// モデルの状態監視を停止
        /// </summary>
        protected virtual void StopModelWatcher()
        {
            if (!isWatcherRunning)
                return;

            watcherCancellationToken?.Cancel();
            isWatcherRunning = false;
        }

        /// <summary>
        /// モデルを読み込む
        /// </summary>
        protected virtual void LoadModel()
        {
            try
            {
                // 古いモデルのアンロード
                UnloadModel();

                // 新しいモデルを読み込む
                if (File.Exists(modelOnnxPath))
                {
                    // ファイルからモデルをロード
                    model = ModelLoader.Load(modelOnnxPath);
                    engine = new Worker(model, BackendType.GPUCompute);
                    isModelLoaded = true;
                    Debug.Log("モデルを正常に読み込みました");
                }
                else
                {
                    Debug.LogError($"モデルファイルが見つかりません: {modelOnnxPath}");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"モデル読み込みエラー: {ex.Message}");
                isModelLoaded = false;
            }
        }

        /// <summary>
        /// モデルをアンロードする
        /// </summary>
        protected virtual void UnloadModel()
        {
            if (engine != null)
            {
                engine.Dispose();
                engine = null;
            }

            if (model != null)
            {
                model = null;
            }

            // GCを促進
            System.GC.Collect();
            isModelLoaded = false;
        }

        /// <summary>
        /// モデルを作成する
        /// </summary>
        public virtual void CreateModel()
        {
            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments =
                        $"{scriptPath} create --shared_folder \"{sharedFolderPath}\""
                        + (
                            string.IsNullOrEmpty(modelModulePath)
                                ? ""
                                : $" --model_module \"{modelModulePath}\""
                        ),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                };

                Process process = new Process { StartInfo = startInfo };
                process.Start();

                // 非同期で出力を読み取る
                process.BeginOutputReadLine();
                process.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Debug.Log($"Python出力: {e.Data}");
                };

                process.BeginErrorReadLine();
                process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Debug.LogError($"Pythonエラー: {e.Data}");
                };

                Debug.Log("モデル作成プロセスを開始しました");
            }
            catch (Exception ex)
            {
                Debug.LogError($"モデル作成プロセスの起動に失敗しました: {ex.Message}");
            }
        }

        /// <summary>
        /// モデルを一度だけトレーニングする
        /// </summary>
        public virtual void TrainModelOnce(
            int epochs = 100,
            int batchSize = 32,
            float learningRate = 0.001f
        )
        {
            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments =
                        $"{scriptPath} train --shared_folder \"{sharedFolderPath}\""
                        + $" --epochs {epochs} --batch_size {batchSize} --lr {learningRate}"
                        + (
                            string.IsNullOrEmpty(modelModulePath)
                                ? ""
                                : $" --model_module \"{modelModulePath}\""
                        ),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                };

                Process process = new Process { StartInfo = startInfo };
                process.Start();

                // 非同期で出力を読み取る
                process.BeginOutputReadLine();
                process.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Debug.Log($"Python出力: {e.Data}");
                };

                process.BeginErrorReadLine();
                process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Debug.LogError($"Pythonエラー: {e.Data}");
                };

                Debug.Log("モデルトレーニングプロセスを開始しました");
            }
            catch (Exception ex)
            {
                Debug.LogError($"モデルトレーニングプロセスの起動に失敗しました: {ex.Message}");
            }
        }

        /// <summary>
        /// 自動学習サーバーを開始する
        /// </summary>
        public virtual void StartTrainingServer(int checkInterval = 10)
        {
            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments =
                        $"{scriptPath} server --shared_folder \"{sharedFolderPath}\""
                        + $" --check_interval {checkInterval}"
                        + (
                            string.IsNullOrEmpty(modelModulePath)
                                ? ""
                                : $" --model_module \"{modelModulePath}\""
                        ),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                };

                Process process = new Process { StartInfo = startInfo };
                process.Start();

                // 非同期で出力を読み取る
                process.BeginOutputReadLine();
                process.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Debug.Log($"Python出力: {e.Data}");
                };

                process.BeginErrorReadLine();
                process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        Debug.LogError($"Pythonエラー: {e.Data}");
                };

                Debug.Log("自動学習サーバーを開始しました");
            }
            catch (Exception ex)
            {
                Debug.LogError($"自動学習サーバーの起動に失敗しました: {ex.Message}");
            }
        }

        /// <summary>
        /// 学習データをCSVに追加
        /// </summary>
        /// <param name="inputData">入力データ</param>
        /// <param name="targetData">目標データ</param>
        public virtual void AddTrainingData(float[] inputData, float[] targetData)
        {
            try
            {
                // ロックファイルの作成
                File.WriteAllText(writeLockPath, DateTime.Now.ToString());

                // CSVファイルの準備
                bool isNewFile = !File.Exists(csvPath);
                using (StreamWriter writer = new StreamWriter(csvPath, true))
                {
                    // 新規ファイルの場合はヘッダーを書き込む
                    if (isNewFile)
                    {
                        List<string> headers = new List<string>();
                        for (int i = 0; i < inputData.Length; i++)
                        {
                            headers.Add($"input_{i}");
                        }
                        for (int i = 0; i < targetData.Length; i++)
                        {
                            headers.Add($"target_{i}");
                        }
                        writer.WriteLine(string.Join(",", headers));
                    }

                    // データ行を書き込む
                    List<string> values = new List<string>();
                    foreach (float val in inputData)
                    {
                        values.Add(val.ToString("F6"));
                    }
                    foreach (float val in targetData)
                    {
                        values.Add(val.ToString("F6"));
                    }
                    writer.WriteLine(string.Join(",", values));
                }

                // ロックファイルの削除
                if (File.Exists(writeLockPath))
                {
                    File.Delete(writeLockPath);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"学習データ追加エラー: {ex.Message}");

                // エラー時にもロックファイルを削除する
                if (File.Exists(writeLockPath))
                {
                    File.Delete(writeLockPath);
                }
            }
        }

        /// <summary>
        /// 最後の予測に対して学習データを追加
        /// </summary>
        /// <param name="targetData">目標データ</param>
        public virtual void AddTrainingDataForLastPrediction(float[] targetData)
        {
            if (lastPrediction != null)
            {
                AddTrainingData(lastPrediction.inputData, targetData);
            }
            else
            {
                Debug.LogWarning("最後の予測データがありません");
            }
        }

        /// <summary>
        /// モデルの状態を取得
        /// </summary>
        protected virtual ModelStatus GetModelStatus()
        {
            try
            {
                if (File.Exists(modelStatusPath))
                {
                    string json = File.ReadAllText(modelStatusPath);
                    return JsonUtility.FromJson<ModelStatus>(json);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"モデル状態読み取りエラー: {ex.Message}");
            }
            return null;
        }

        /// <summary>
        /// モデルの状態を更新
        /// </summary>
        protected virtual void UpdateModelStatus(string status, string errorMessage = null)
        {
            try
            {
                ModelStatus statusData = new ModelStatus
                {
                    status = status,
                    timestamp = DateTime.Now.Ticks,
                    error = errorMessage,
                };

                string json = JsonUtility.ToJson(statusData);
                File.WriteAllText(modelStatusPath, json);
            }
            catch (Exception ex)
            {
                Debug.LogError($"モデル状態更新エラー: {ex.Message}");
            }
        }

        /// <summary>
        /// モデル設定を取得
        /// </summary>
        protected virtual ModelConfig GetModelConfig()
        {
            try
            {
                if (File.Exists(configPath))
                {
                    string json = File.ReadAllText(configPath);
                    return JsonUtility.FromJson<ModelConfig>(json);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"モデル設定読み取りエラー: {ex.Message}");
            }
            return null;
        }
    }

    /// <summary>
    /// 入力データと出力の履歴を保持するクラス
    /// </summary>
    [Serializable]
    public class InputData
    {
        public float[] inputData;
        public float[] outputData;
        public DateTime timestamp;
    }

    /// <summary>
    /// モデルの状態を表すクラス
    /// </summary>
    [Serializable]
    public class ModelStatus
    {
        public string status;
        public long timestamp;
        public string error;
    }

    /// <summary>
    /// モデル設定を表すクラス
    /// </summary>
    [Serializable]
    public class ModelConfig
    {
        public InputOutputConfig input;
        public InputOutputConfig output;
    }

    /// <summary>
    /// 入出力設定を表すクラス
    /// </summary>
    [Serializable]
    public class InputOutputConfig
    {
        public string name;
        public int[] shape;
        public Dictionary<string, string> dynamic_axes;
    }

    /// <summary>
    /// メインスレッドでの実行をサポートするヘルパークラス
    /// </summary>
    public class UnityMainThreadDispatcher : MonoBehaviour
    {
        private static UnityMainThreadDispatcher _instance;
        private readonly Queue<Action> _executionQueue = new Queue<Action>();
        private readonly object _lock = new object();

        public static UnityMainThreadDispatcher Instance
        {
            get
            {
                if (_instance == null)
                {
                    var go = new GameObject("UnityMainThreadDispatcher");
                    _instance = go.AddComponent<UnityMainThreadDispatcher>();
                    DontDestroyOnLoad(go);
                }
                return _instance;
            }
        }

        private void Awake()
        {
            if (_instance == null)
            {
                _instance = this;
                DontDestroyOnLoad(gameObject);
            }
        }

        private void Update()
        {
            lock (_lock)
            {
                while (_executionQueue.Count > 0)
                {
                    _executionQueue.Dequeue().Invoke();
                }
            }
        }

        public void Enqueue(Action action)
        {
            lock (_lock)
            {
                _executionQueue.Enqueue(action);
            }
        }
    }
}
