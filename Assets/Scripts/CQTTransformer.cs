using Unity.Sentis;
using System;
using UnityEngine;
using UnityEngine.Serialization;
using System.Numerics;
using System.Collections.Generic;

namespace Ext
{
    public class CQTTransformer : MonoBehaviour
    {
        [SerializeField] private ModelAsset _modelAsset;
        [SerializeField] private int _bufferLength;

        public int bufferLength => _bufferLength;

        private Model _runtimeModel;
        private Tensor<float> _inputTensor;
        private float[] _internalData;
        private double[,] _cqtMagData;
        private Complex[,] _cqtResult;
        private int stackSize;
        private float[] stackedData;
        private float[] outputBuffer;
        private ConstantQTransform _cqt;
        private Worker _worker;
        private Queue<bool> stackedQueue = new();

        public void Initialize(int stacks)
        {
            _runtimeModel = ModelLoader.Load(_modelAsset);
            _worker = new Worker(_runtimeModel, BackendType.GPUCompute);
            _cqt = new ConstantQTransform();
            _cqtResult = _cqt.CreateResultBuffer(_bufferLength);
            _cqtMagData = Signals.CreateDoubleRawData(_cqtResult);
            _internalData = Signals.CreateFloatRawData(_cqtResult);
            outputBuffer = _cqt.CreateOutputBuffer();
            stackSize = _cqtResult.GetLength(0);
            stackedData = new float[stackSize * stacks];
            _inputTensor = new Tensor<float>(new TensorShape(1, stackSize * stacks, _cqtResult.GetLength(1)), clearOnInit: true);
        }

        public void SetInput(float[] x)
        {
            _cqt.Transform(x, _cqtResult);
            Signals.StdNorm(_cqtResult, _cqtMagData, _internalData);
            Array.Copy(stackedData, stackSize, stackedData, 0, stackedData.Length - stackSize);
            Array.Copy(_internalData, 0, stackedData, stackedData.Length - stackSize, stackSize);
            _inputTensor.Upload(stackedData);
            _worker.SetInput("input", _inputTensor);
            _worker.Schedule();
            stackedQueue.Enqueue(true);
        }

        async void Update()
        {
            if (stackedQueue.Count > 0)
            {
                stackedQueue.Dequeue();
                var outputTensor = _worker.PeekOutput() as Tensor<float>;
                var cpuCopyTensor = await outputTensor.ReadbackAndCloneAsync();
                outputBuffer = cpuCopyTensor.DownloadToArray();
                Debug.Log($"Output tensor value {string.Join(',', outputBuffer)}");
                cpuCopyTensor.Dispose();
            }
        }

        public float[] GetLastOutput() => outputBuffer;

        private void OnDestroy()
        {
            _inputTensor.Dispose();
            _worker.Dispose();
        }
    }
}