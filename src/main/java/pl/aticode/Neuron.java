package pl.aticode;

import lombok.Getter;
import lombok.Setter;

import java.util.Random;

public class Neuron {

    private static final double LAMBDA = 1.0507;
    private static final double ALPHA = 1.67326;
    @Getter @Setter private double bias;
    @Getter @Setter private double output;
    @Getter @Setter private double error;
    @Getter private final double [] weightArray;
    @Getter private final double [] previousWeightUpdateArray;
    private final ActivationType activationType;


    public Neuron(int weight, ActivationType activationType, Random random, double bias) {
        this.activationType = activationType;
        this.bias = bias;
        this.weightArray = new double[weight];
        this.previousWeightUpdateArray = new double[weight];
        for (int i = 0; i < this.weightArray.length; i++) {
            this.weightArray[i] = random.nextDouble(-0.8, 0.8);
            this.previousWeightUpdateArray[i] = 0.0;
        }
    }


    public void activation(double in) throws Exception {
        switch (this.activationType) {
            case SIGMOID -> this.output = sigmoid(in);
            case HARDSIGMOID -> this.output = hardSigmoid(in);
            case SELU -> this.output = selu(in);
            case RELU -> this.output = relu(in);
            case SIN -> this.output = sin(in);
            case TANGENT -> this.output = tangent(in);
            case TANH -> this.output = tanh(in);
            case SOFTPLUS -> this.output = softPlus(in);
            default -> throw new Exception("Lack of activation type!");
        }
    }

    public double derivative() throws Exception {
        switch (this.activationType) {
            case SIGMOID -> {
                return sigmoidDerivative(this.output);
            }
            case HARDSIGMOID -> {
                return hardSigmoidDerivative(this.output);
            }
            case SELU -> {
                return seluDerivative(this.output);
            }
            case RELU -> {
                return reluDerivative(this.output);
            }
            case SIN -> {
                return sinDerivative(this.output);
            }
            case TANGENT -> {
                return tangentDerivative(this.output);
            }
            case TANH -> {
                return tanhDerivative(this.output);
            }
            case SOFTPLUS -> {
                return softPlusDerivative(this.output);
            }
            default -> throw new Exception("Lack of activation type!");
        }
    }
    private double sigmoid(double in) {
        return 1 / (1 + Math.exp(-in));
    }

    private double sigmoidDerivative(double in) {
        return in * (1 - in);
    }

    private double hardSigmoid(double in) {
        if (in <= -3) {
            return 0;
        }
        if (in >= 3) {
            return 1;
        }
        return in / 6 + 0.5;
    }

    private double hardSigmoidDerivative(double in) {
        if (in >= 3 || in <= -3) {
            return 0;
        }
        return 1.0 / 6;
    }

    private double tanh(double in) {
        return Math.tanh(in);
    }

    private double tanhDerivative(double in) {
        return 1 / (Math.cosh(in) * Math.cosh(in));
    }

    private double tangent(double in) {
        return Math.tan(in);
    }

    private double tangentDerivative(double in) {
        return 1 / (Math.cos(in) * Math.cos(in));
    }

    private double sin(double in) {
        return Math.sin(in);
    }

    private double sinDerivative(double in) {
        return Math.cos(in);
    }

    private double selu(double x) {
        if (x > 0) {
            return LAMBDA * x;
        }
        return LAMBDA * ALPHA * (Math.exp(x) - 1);
    }

    private double seluDerivative(double x) {
        if (x > 0) {
            return LAMBDA;
        }
        return LAMBDA * ALPHA * Math.exp(x);
    }

    private double relu(double x) {
        if (x > 0) {
            return x;
        }
        return 0;
    }

    private double reluDerivative(double x) {
        if (x > 0) {
            return 1;
        }
        return 0;
    }

    private double softPlus(double x) {
        return Math.log(1 + Math.exp(x));
    }

    private double softPlusDerivative(double x) {
        return sigmoid(x);
    }
}
