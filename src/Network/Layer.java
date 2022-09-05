package Network;

import Util.Helper;

import java.util.ArrayList;
import java.util.List;

public class Layer {

    public List<Neuron> neurons;
    public double totalDelta;

    //First Layer
    public Layer(double[] inputs) {
        neurons = new ArrayList<>();
        for(double input: inputs) {
            neurons.add(new Neuron(input));
        }
    }

    //Hidden and Last Layers
    public Layer(int incomingNeurons, int neuronLength) {
        neurons = new ArrayList<>();

        //Generate weights for each neuron in previous layer connecting to each neuron in current layer
        for(int i = 0; i < neuronLength; i++) {
            double[] weights = new double[incomingNeurons];

            for(int j = 0; j < incomingNeurons; j++) {
                weights[j] = Helper.getRandomDouble();
            }

            neurons.add(new Neuron(weights, Helper.getRandomDouble()));
        }

    }

}
