package pl.aticode;

import lombok.Getter;

import java.util.Random;

public class OutputLayer {

    @Getter private Neuron[] lastHiddenNeuronArray;
    @Getter private Neuron[] neuronArray;

    private OutputLayer() {
    }

    public static class Builder {

        private final OutputLayer outputLayer;
        private final Random random;

        public Builder() {
            this.outputLayer = new OutputLayer();
            this.random = new Random();
        }

        public Builder withNeurons(int neuronOut, ActivationType activationType) {
            double bias = this.random.nextDouble(-0.5, 0.5);
            this.outputLayer.neuronArray = new Neuron[neuronOut];
            for (int i = 0; i < this.outputLayer.neuronArray.length; i++) {
                this.outputLayer.neuronArray[i] = new Neuron(this.outputLayer.lastHiddenNeuronArray.length, activationType, this.random, bias);
            }
            return this;
        }

        public Builder withHiddenLayers(HiddenLayers hiddenLayers) {
            int layers = hiddenLayers.getLayerList().size();
            this.outputLayer.lastHiddenNeuronArray = hiddenLayers.getLayerList().get(layers - 1);
            return this;
        }

        public OutputLayer build() {
            return this.outputLayer;
        }
    }
}
