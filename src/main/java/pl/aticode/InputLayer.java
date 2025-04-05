package pl.aticode;

import lombok.Getter;

import java.util.Random;

public class InputLayer {

    @Getter private Neuron[] neuronArray;

    private InputLayer() {
    }

    public static class Builder {

        private final InputLayer inputLayer;
        private final Random random;

        public Builder() {
            this.inputLayer = new InputLayer();
            this.random = new Random();
        }

        public Builder withNeurons(int neuronIn, ActivationType activationType) {
            double bias = this.random.nextDouble(-0.5, 0.5);
            this.inputLayer.neuronArray = new Neuron[neuronIn];
            for (int i = 0; i < this.inputLayer.neuronArray.length; i++) {
                this.inputLayer.neuronArray[i] = new Neuron(1, activationType, this.random, bias);
            }
            return this;
        }

        public InputLayer build() {
            return this.inputLayer;
        }
    }
}
