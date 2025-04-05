package pl.aticode;

import lombok.Getter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class HiddenLayers {

    @Getter private InputLayer inputLayer;
    @Getter private List<Neuron[]> layerList;

    private HiddenLayers() {
    }

    public static class Builder {

        private final HiddenLayers hiddenLayers;
        private final Random random;

        public Builder() {
            this.hiddenLayers = new HiddenLayers();
            this.hiddenLayers.layerList = new ArrayList<>();
            this.random = new Random();
        }

        public Builder addNeuronLayer(int neurons, ActivationType activationType) {
            double bias = this.random.nextDouble(-0.5, 0.5);
            Neuron[] neuronArray = new Neuron[neurons];
            for (int i = 0; i < neuronArray.length; i++) {
                if (this.hiddenLayers.layerList.isEmpty()) {
                    neuronArray[i] = new Neuron(this.hiddenLayers.inputLayer.getNeuronArray().length, activationType, this.random, bias);
                } else {
                    int hiddenLayerListSize = this.hiddenLayers.getLayerList().size();
                    neuronArray[i] = new Neuron(this.hiddenLayers.layerList.get(hiddenLayerListSize -  1).length, activationType, this.random, bias);
                }
            }
            this.hiddenLayers.layerList.add(neuronArray);
            return this;
        }

        public Builder withInputLayer(InputLayer inputLayer) {
            this.hiddenLayers.inputLayer = inputLayer;
            return this;
        }

        public HiddenLayers build() {
            return this.hiddenLayers;
        }
    }
}
