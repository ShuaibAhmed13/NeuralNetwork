package Network;

import Util.Helper;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    public List<Layer> layers;

    public NeuralNetwork() {
        layers = new ArrayList<>();
    }

    public void addLayer(double[] inputs) {
        layers.add(new Layer(inputs));
    }

    public void addLayer(int inputConnections, int neuronAmount) {
        layers.add(new Layer(inputConnections, neuronAmount));
    }

    public void forward() {
        //Iterate through layers starting at layer 1 because layer 0 has input values
        for(int i = 1; i < layers.size(); i++) {
            //Iterate through neurons at current layer
            for(int j = 0; j < layers.get(i).neurons.size(); j++) {
                double sum = 0;
                //Iterate through the weights of each neuron
                for(int k = 0; k < layers.get(i).neurons.get(j).weights.length; k++) {
                    //Add the product of the respective weight x respective neuron value (from previous layer)
                    //to the sum
                    sum += layers.get(i).neurons.get(j).weights[k] * layers.get(i-1).neurons.get(k).value;
                }
                //Add the bias and set current neuron value to the sum after it is passed through the sigmoid function
                sum += layers.get(i).neurons.get(j).bias;
                layers.get(i).neurons.get(j).value = Helper.sigmoid(sum);
            }
        }
    }

    public void backPropagation() {

    }
}
