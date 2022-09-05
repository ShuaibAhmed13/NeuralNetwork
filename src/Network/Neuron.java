package Network;

public class Neuron {

    public double value;
    public double weightedValue;
    public double[] weights;
    public double[] cacheWeights;
    public double bias;
    public double cacheBias;
    public double gradient;


    //First Layer Neuron
    public Neuron(double input) {
        this.value = input;
        this.bias = 0;
    }

    //Hidden and Output Layer Neurons
    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
        this.cacheWeights = new double[weights.length];
    }

}
