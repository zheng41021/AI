/**
 * Perceptron 感知机
 * 是一种最简单的人工神经元模型，用于二分类问题的预测。
 * 它模拟了神经元之间的连接和信号传递方式，
 * 通过输入数据和权重的线性组合以及一个阈值函数来进行预测。
 */ 
public class Perceptron {

    private double[] weights; // 权重
    private double bias; // 偏置（bias）是一个与输入特征无关的常数，用于调整模型的预测结果。

    public Perceptron(int inputSize) {
        weights = new double[inputSize];
        initializeWeights();
        bias = 0.0;
    }

    /**
     * 对感知机（perceptron）的权重进行随机初始化。在感知机的训练过程中，
     * 权重是非常重要的模型参数，它们决定了输入特征对预测结果的影响程度。
     * 在初始化阶段，权重的初始值通常是随机选择的。这是因为在训练开始时，
     * 我们对数据集的分布和模式了解不多，因此随机初始化可以帮助模型在初始阶段探索不同的权重组合。
     * 随机初始化权重意味着为每个特征与其对应的权重分配一个随机的初始值。
     * 这些初始值可以是从某个分布中随机抽取的，例如均匀分布或高斯分布。
     * 通过随机初始化权重，感知机可以开始训练并学习调整权重，
     * 以适应输入数据和期望输出之间的关系。
     */
    private void initializeWeights() {
        for (int i = 0; i < weights.length; i++) {
            // initializes the weight to a random value
            weights[i] = Math.random();
        }
    }

    /**
     * 感知机模型的预测方法
     * 给定输入特征数组，计算输入特征与对应权重的加权和，并加上偏置项。
     * 然后，使用阈值函数对加权和进行判断，并返回预测结果。
     */
    public int predict(double[] inputs) {
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        sum += bias;

        // 使用阈值函数进行预测，这里使用简单的阈值函数（大于等于0为正类，小于0为负类）
        return (sum >= 0) ? 1 : -1;
    }

    /**
     * 训练机器学习模型
     * 训练数据是指用于模型训练的输入特征和对应的输出标签。
     * 通过提供训练数据，我们可以让模型基于这些数据来学习输入特征与输出标签之间的关系和模式。
     */
    public static void main(String[] args) {
        // 创建一个2维输入向量的感知器模型
        Perceptron perceptron = new Perceptron(2);

        // 定义训练集
        double[][] trainingInputs = { { 0, 1 }, { 1, 0 }, { 0, 0 }, { 1, 1 } };

        // 标签，即实际结果。非预测结果，根据实际结果指导权重和偏置的值进行改变，从而预测输入特征数据。
        int[] labels = { -1, -1, -1, 1 };

        // 训练感知器模型
        for (int epoch = 0; epoch < 10; epoch++) { // 在机器学习和深度学习中，epoch（时代）是指将整个训练数据集从头到尾迭代一次的次数。每个epoch包含了对整个训练数据集的一次完整遍历。
            for (int i = 0; i < trainingInputs.length; i++) {
                int prediction = perceptron.predict(trainingInputs[i]);
                int label = labels[i];
                if (prediction != label) {
                    // 更新权重和偏差
                    for (int j = 0; j < perceptron.weights.length; j++) {
                        perceptron.weights[j] += label * trainingInputs[i][j];
                    }
                    perceptron.bias += label;
                }
            }
        }

        // 测试感知器模型
        double[] testInput = { 1, 1 }; // 训练数据中存在的数据，可以证明在充足的学习数据下，预测结果符合实际结果。
        int prediction = perceptron.predict(testInput);
        System.out.println("Prediction: " + prediction);

        testInput[0] = 0; // 训练数据中存在的数据，可以证明在充足的学习数据下，预测结果符合实际结果。
        prediction = perceptron.predict(testInput);
        System.out.println("Prediction: " + prediction);

        testInput[0] = -100; // 训练数据中不存在的数据，根据预测模型，得出负类的预测结果
        prediction = perceptron.predict(testInput);
        System.out.println("Prediction: " + prediction);

        testInput[0] = 100; // 训练数据中不存在的数据，根据预测模型，得出预测结果
        prediction = perceptron.predict(testInput);
        System.out.println("Prediction: " + prediction);

    }
}