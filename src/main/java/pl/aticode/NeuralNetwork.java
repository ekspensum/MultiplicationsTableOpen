package pl.aticode;

import lombok.Getter;

import java.util.Collection;
import java.util.List;

public class NeuralNetwork {
    private static final String LOSS_THRESHOLD_PREFIX = "LOSS_THRESHOLD_";
    private static final double REDUCTION_RATE = 0.93;
    private final int epochs;
    private double learnFactor;
    private int learnFactorLevel;
    private final double momentum;
    private final double lossThreshold;
    private final InputLayer inputLayer;
    private final HiddenLayers hiddenLayers;
    @Getter private final OutputLayer outputLayer;
    private double meanSquareLossPrevious;
    private int errorGrowthCounter;

    public NeuralNetwork(int epochs, double learnFactor, double momentum, double lossThreshold) {
        this.epochs = epochs;
        this.learnFactor = learnFactor;
        this.momentum = momentum;
        this.learnFactorLevel = 1;
        this.lossThreshold = lossThreshold;
        this.meanSquareLossPrevious = 1.0;
        this.inputLayer = new InputLayer.Builder()
                .withNeurons(2, ActivationType.SIGMOID)
                .build();
        this.hiddenLayers = new HiddenLayers.Builder()
                .withInputLayer(inputLayer)
                .addNeuronLayer(50, ActivationType.SIGMOID)
                .addNeuronLayer(10, ActivationType.SIGMOID)
                .build();
        this.outputLayer = new OutputLayer.Builder()
                .withHiddenLayers(hiddenLayers)
                .withNeurons(1, ActivationType.SELU)
                .build();
    }

    public int train(Collection<TrainData> trainDataSet) throws Exception {
        int epoch;
        for (epoch = 0; epoch <= this.epochs; epoch++) {
            for (TrainData trainData : trainDataSet) {
                double [] inputArray = new double [] {trainData.getMultiplier1(), trainData.getMultiplier2()};
                double[] outputArray = this.feedForward(inputArray);
                trainData.setPredictResultArray(outputArray);
                this.backPropagation(trainData);
            }
            double meanSquareLoss = Util.meanSquareLoss(trainDataSet);
            if (epoch % 10000 == 0) {
                System.out.printf("Epoch: %s | Loss: %.10f%n", epoch, meanSquareLoss);
                createLearningRateReduction(epoch, meanSquareLoss);
                meanSquareLossWarningAndLearnFactorReduce(meanSquareLoss);
            }
            if (meanSquareLoss < this.lossThreshold) {
                break;
            }
        }
        return epoch;
    }

    public void computeInputLayer(double[] inputArray) throws Exception {
        Neuron[] neuronArray = this.inputLayer.getNeuronArray();
        for (int i = 0; i < neuronArray.length; i++) {
            double preActivation = 0;
            for (double weight : neuronArray[i].getWeightArray()) {
                preActivation = weight * inputArray[i];
            }
            neuronArray[i].activation(preActivation + neuronArray[i].getBias());
        }
    }

    public void computeHiddenLayers() throws Exception {
        List<Neuron[]> hiddenLayerList = this.hiddenLayers.getLayerList();
        Neuron[] inputNeurons = this.hiddenLayers.getInputLayer().getNeuronArray();
        for (int i = 0; i < hiddenLayerList.size(); i++) {
            for (Neuron hiddenNeuron : hiddenLayerList.get(i)) {
                Neuron[] hiddenNeuronsPreviousLayer = null;
                if (i > 0) {
                    hiddenNeuronsPreviousLayer = hiddenLayerList.get(i - 1);
                }
                double preActivation = 0;
                for (int j = 0; j < hiddenNeuron.getWeightArray().length; j++) {
                    if (i == 0) {
                        preActivation += hiddenNeuron.getWeightArray()[j] * inputNeurons[j].getOutput();
                    } else {
                        preActivation += hiddenNeuron.getWeightArray()[j] * hiddenNeuronsPreviousLayer[j].getOutput();
                    }
                }
                hiddenNeuron.activation(preActivation + hiddenNeuron.getBias());
            }
        }
    }

    public double[] computeOutputLayer() throws Exception {
        double[] outputArray = new double[this.outputLayer.getNeuronArray().length];
        Neuron[] lastHiddenNeuronArray = this.outputLayer.getLastHiddenNeuronArray();
        Neuron[] outputNeurons = this.outputLayer.getNeuronArray();
        for (int i = 0; i < outputNeurons.length; i++) {
            double preActivation = 0;
            for (int j = 0; j < lastHiddenNeuronArray.length; j++) {
                preActivation += lastHiddenNeuronArray[j].getOutput() * outputNeurons[i].getWeightArray()[j];
            }
            outputNeurons[i].activation(preActivation + outputNeurons[i].getBias());
            outputArray[i] = outputNeurons[i].getOutput();
        }
        return outputArray;
    }

    public double[] feedForward(double[] inputArray) throws Exception {
        computeInputLayer(inputArray);
        computeHiddenLayers();
        return computeOutputLayer();
    }

    private void backPropagation(TrainData trainData) throws Exception {
        double[] unitErrorArray = new double[trainData.getPredictResultArray().length];
        for (int i = 0; i < trainData.getPredictResultArray().length; i++) {
            unitErrorArray[i] = trainData.getActualResult() - trainData.getPredictResultArray()[i];
        }
        Neuron[] lastHiddenNeuronArray = this.outputLayer.getLastHiddenNeuronArray();
        Neuron[] outputNeuronArray = this.outputLayer.getNeuronArray();
        for (int on_idx = 0; on_idx < outputNeuronArray.length; on_idx++) {
            outputNeuronArray[on_idx].setError(unitErrorArray[on_idx] * outputNeuronArray[on_idx].derivative());
            outputNeuronArray[on_idx].setBias(outputNeuronArray[on_idx].getBias() + outputNeuronArray[on_idx].getError() * this.learnFactor);
            double[] weightArrayOutput = outputNeuronArray[on_idx].getWeightArray();
            double[] previousWeightUpdateArrayOutput = outputNeuronArray[on_idx].getPreviousWeightUpdateArray();
            for (int i = 0; i < weightArrayOutput.length; i++) {
                double weightUpdateOutput = outputNeuronArray[on_idx].getError() * lastHiddenNeuronArray[i].getOutput() * this.learnFactor + this.momentum * previousWeightUpdateArrayOutput[i];
                weightArrayOutput[i] = weightArrayOutput[i] + weightUpdateOutput;
                previousWeightUpdateArrayOutput[i] = weightUpdateOutput;
            }
        }

        Neuron[] inputNeurons = this.hiddenLayers.getInputLayer().getNeuronArray();
        List<Neuron[]> hiddenLayerList = this.hiddenLayers.getLayerList();
        int hiddenLayerSize = hiddenLayerList.size();
        for (int hl_idx = hiddenLayerSize; hl_idx >= 1 ; hl_idx--) {
            Neuron[] hiddenNeuronArrayCurrent = hiddenLayerList.get(hl_idx - 1);
            for (int i = 0; i < hiddenNeuronArrayCurrent.length; i++) {
                if (hiddenLayerSize == hl_idx) {
                    calculateNeuronError(outputNeuronArray, hiddenNeuronArrayCurrent, i);
                } else {
                    Neuron[] hiddenNeuronArrayNext = hiddenLayerList.get(hl_idx);
                    calculateNeuronError(hiddenNeuronArrayNext, hiddenNeuronArrayCurrent, i);
                }
                hiddenNeuronArrayCurrent[i].setBias(hiddenNeuronArrayCurrent[i].getBias() + hiddenNeuronArrayCurrent[i].getError() * this.learnFactor);
                double[] weightArrayOfCurrentHiddenLayer = hiddenNeuronArrayCurrent[i].getWeightArray();
                double[] previousWeightUpdateArrayHidden = hiddenNeuronArrayCurrent[i].getPreviousWeightUpdateArray();
                for (int j = 0; j < weightArrayOfCurrentHiddenLayer.length; j++) {
                    if (hl_idx == 1) {
                        double weightUpdateHiddenInput = hiddenNeuronArrayCurrent[i].getError() * inputNeurons[j].getOutput() * this.learnFactor + this.momentum * previousWeightUpdateArrayHidden[j];
                        weightArrayOfCurrentHiddenLayer[j] = weightArrayOfCurrentHiddenLayer[j] + weightUpdateHiddenInput;
                        previousWeightUpdateArrayHidden[j] = weightUpdateHiddenInput;
                    } else {
                        Neuron[] hiddenNeuronArrayPrevious = hiddenLayerList.get(hl_idx - 2);
                        double weightUpdateHidden = hiddenNeuronArrayCurrent[i].getError() * hiddenNeuronArrayPrevious[j].getOutput() * this.learnFactor + this.momentum * previousWeightUpdateArrayHidden[j];
                        weightArrayOfCurrentHiddenLayer[j] = weightArrayOfCurrentHiddenLayer[j] + weightUpdateHidden;
                        previousWeightUpdateArrayHidden[j] = weightUpdateHidden;
                    }
                }
            }
        }

        Neuron[] firstHiddenNeuronArray = hiddenLayerList.get(0);
        for (int i = 0; i < inputNeurons.length; i++) {
            calculateNeuronError(firstHiddenNeuronArray, inputNeurons, i);
            inputNeurons[i].setBias(inputNeurons[i].getBias() + inputNeurons[i].getError() * this.learnFactor);
            double[] weightArrayInput = inputNeurons[i].getWeightArray();
            double[] previousWeightUpdateArrayInput = inputNeurons[i].getPreviousWeightUpdateArray();
            for (int j = 0; j < weightArrayInput.length; j++) {
                double weightUpdateInput = inputNeurons[i].getError() * this.learnFactor + this.momentum * previousWeightUpdateArrayInput[j];
                weightArrayInput[j] = weightArrayInput[j] + weightUpdateInput;
                previousWeightUpdateArrayInput[j] = weightUpdateInput;
            }
        }
    }

    private void calculateNeuronError(Neuron[] neuronArrayNext, Neuron[] neuronArrayCurrent, int i) throws Exception {
        double hiddenNeuronError = 0;
        for (Neuron hiddenNeuron : neuronArrayNext) {
            hiddenNeuronError += hiddenNeuron.getWeightArray()[i] * hiddenNeuron.getError();
        }
        double hiddenNeuronErrorAvg = hiddenNeuronError / neuronArrayNext.length;
        neuronArrayCurrent[i].setError(hiddenNeuronErrorAvg * neuronArrayCurrent[i].derivative());
    }

    private void createLearningRateReduction(int epoch, double meanSquareLoss) {
        for (int level = 1; level < LossThresholds.values().length; level++) {
            if (meanSquareLoss < LossThresholds.valueOf(LOSS_THRESHOLD_PREFIX + level).getValue() && this.learnFactorLevel == level) {
                this.learnFactor = Util.round(this.learnFactor * REDUCTION_RATE, 5);
                this.learnFactorLevel = level + 1;
                System.out.printf("Threshold-" + level + ": learning rate: %.6f | epoch: %s | Loss: %.12f%n", this.learnFactor, epoch, meanSquareLoss);
                break;
            }
        }
    }

    private void meanSquareLossWarningAndLearnFactorReduce(double meanSquareLoss) {
        if (this.meanSquareLossPrevious < meanSquareLoss) {
            System.out.println("WARNING - mean square loss growth");
            this.errorGrowthCounter++;
            if (this.errorGrowthCounter > 4) {
                this.learnFactor = Util.round(this.learnFactor * REDUCTION_RATE, 5);
                System.out.printf("Learning rate was reduced: %.6f%n", this.learnFactor);
                this.errorGrowthCounter = 0;
            }
        } else {
            this.errorGrowthCounter = 0;
        }
        this.meanSquareLossPrevious = meanSquareLoss;
    }

}
