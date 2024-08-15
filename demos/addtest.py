import numpy as np
import torch

from metabci.brainda.algorithms.decomposition.base import AdvancedSignalProcessing
from metabci.brainda.algorithms.deep_learning import CNN_GRU_Attn
from metabci.brainda.algorithms.feature_analysis.visualization import MetaBCIVisualization
from metabci.brainflow.workers import EnhancedProcessWorker

# 测试 AdvancedSignalProcessing 类
def test_advanced_signal_processing():
    print("Testing AdvancedSignalProcessing...")
    # 生成64通道1000个数据点的随机信号作为测试数据
    data = np.random.rand(64, 1000)  # 64个通道的示例数据
    fs = 250  # 采样率
    reference_signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # 参考信号

    advanced_signal_processor = AdvancedSignalProcessing(data, fs)
    
    # 自适应滤波
    filtered_data_adaptive = advanced_signal_processor.adaptive_filter(reference_signal)
    print("Adaptive filter output shape:", filtered_data_adaptive.shape)

    # 波形分解
    coeffs = advanced_signal_processor.wavelet_decomposition()
    print("Wavelet decomposition coefficients length:", len(coeffs))

    # ICA重构
    reconstructed_data = advanced_signal_processor.ica_reconstruction()
    print("ICA reconstructed data shape:", reconstructed_data.shape)

    # 伪迹标记
    artifact_indices = advanced_signal_processor.mark_artifacts(threshold=100)
    print("Artifact indices:", artifact_indices)

    # 基于稀疏表示的滤波
    reconstructed_signal_sparse = advanced_signal_processor.apply_sparse_filtering()
    print("Sparse reconstructed signal shape:", reconstructed_signal_sparse.shape)


    # 非线性滤波
    filtered_data_median = advanced_signal_processor.apply_median_filter()
    print("Median filtered data shape:", filtered_data_median.shape)

# 测试 CNN_GRU_Attn 模型
def test_cnn_gru_attn():
    print("Testing CNN_GRU_Attn...")
    # 生成10个样本的随机数据，每个样本有16个通道，每个通道有100个采样点
    X = np.random.rand(10, 16, 100)  # [batch size, number of channels, number of sample points]
    y = np.random.randint(0, 2, size=(10,))  # 二分类标签

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    model = CNN_GRU_Attn(n_channels=16, n_samples=100, n_classes=2)
    
    # 测试前向传播
    outputs = model(X)
    print("CNN_GRU_Attn model output shape:", outputs.shape)

# 测试 MetaBCIVisualization 类
def test_meta_bci_visualization():
    print("Testing MetaBCIVisualization...")
    # 生成16个通道1000个数据点的随机信号作为测试数据
    data = np.random.rand(16, 1000)  # 16个通道的示例数据
    fs = 250  # 采样率

    visualization = MetaBCIVisualization(data, fs)

    # 频谱密度图
    visualization.plot_power_spectral_density()

    # 伪彩色图
    visualization.plot_pseudocolor()

    # 时频分析
    visualization.plot_time_frequency()

    # 混淆矩阵和分类评估
    y_true = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)
    visualization.evaluate_classification(y_true, y_pred, labels=["Class 0", "Class 1"])

# 测试 EnhancedProcessWorker 类
def test_enhanced_process_worker():
    print("Testing EnhancedProcessWorker...")
    # 生成1000个数据点64个通道的随机信号作为测试数据
    data = np.random.rand(1000, 64)  # 假设有64个通道的数据

    worker = EnhancedProcessWorker(timeout=1, name='test_worker', fs=250)

    # 数据处理
    worker.pre()
    worker.consume(data)
    worker.post()
    print("EnhancedProcessWorker test completed.")

# 测试所有功能
if __name__ == "__main__":

    test_advanced_signal_processing()
    test_cnn_gru_attn()
    test_meta_bci_visualization()
    test_enhanced_process_worker()
    print("All tests completed.")



