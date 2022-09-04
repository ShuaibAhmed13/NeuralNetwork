package Network;

public class Neuron {

    public double value;
    public double[] weights;
    public double bias;
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
    }

}
