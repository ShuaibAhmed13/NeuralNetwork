package Network;

import Training.TrainingObject;
import Util.Helper;

import java.util.ArrayList;
import java.util.Arrays;
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

    public void forward(double[] inputs) {
        Layer newLayer = new Layer(inputs);
        layers.set(0, newLayer);
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
                layers.get(i).neurons.get(j).weightedValue = sum;
                layers.get(i).neurons.get(j).value = Helper.sigmoid(sum);
            }
        }
    }

    public void backPropagation(double learningRate, double expected) {
        int currentLayerIndex = layers.size()-1;
        //Change in cost over change in activated value
        double cost_activatedVal;
        //Change in activated value over change in weighted value
        double activatedVal_weightedVal;
        //Change in weighted value over change in specific weight
        double weightedVal_weight;
        //Add up the total delta of all neurons within a layer
        double totalDelta = 0;
        for(int i = 0; i < layers.get(currentLayerIndex).neurons.size(); i++) {
            Neuron neuron = layers.get(currentLayerIndex).neurons.get(i);
            cost_activatedVal = Helper.qcfDerivative(expected, neuron.value);
            activatedVal_weightedVal = Helper.sigmoidDerivative(neuron.weightedValue);
            double delta = cost_activatedVal * activatedVal_weightedVal;
            totalDelta += delta;
//            layers.get(currentLayerIndex).neurons.get(i).gradient = delta;
            for(int j = 0; j < layers.get(currentLayerIndex).neurons.get(i).weights.length; j++) {
                //Get the value of the neuron that corresponds to the current weight
                double prevActivatedValue = layers.get(currentLayerIndex-1).neurons.get(j).value;
                //totalCost is the change in cost with respect to the change in the current weight
                double totalCost = delta * prevActivatedValue;
                layers.get(currentLayerIndex).neurons.get(i).cacheWeights[j] =
                        layers.get(currentLayerIndex).neurons.get(i).weights[j] - totalCost * learningRate;
            }
            layers.get(currentLayerIndex).neurons.get(i).cacheBias =
                    layers.get(currentLayerIndex).neurons.get(i).bias - delta * learningRate;
        }
        layers.get(currentLayerIndex).totalDelta = totalDelta;

        for(int i = currentLayerIndex-1; i > 0; i--) {
            totalDelta = 0;
            for(int j = 0; j < layers.get(i).neurons.size(); j++) {
                Neuron neuron = layers.get(i).neurons.get(j);
                activatedVal_weightedVal = Helper.sigmoidDerivative(neuron.weightedValue);
                double delta = activatedVal_weightedVal * layers.get(i+1).totalDelta;
                totalDelta += delta;
                for(int k = 0; k < layers.get(i).neurons.get(j).weights.length; k++){
                    double prevActivatedValue = layers.get(i-1).neurons.get(k).value;
                    double totalCost = delta * prevActivatedValue;
                    layers.get(i).neurons.get(j).cacheWeights[k] = layers.get(i).neurons.get(j).weights[k] -
                            totalCost * learningRate;
                }
                layers.get(i).neurons.get(j).cacheBias = layers.get(i).neurons.get(j).bias -
                        delta * learningRate;
            }
            layers.get(i).totalDelta = totalDelta;
        }

        //Iterate through every layer and update the weights and biases
        for(int i = 0; i < layers.size(); i++) {
            for(int j = 0; j < layers.get(i).neurons.size(); j++) {
                layers.get(i).neurons.get(j).weights = layers.get(i).neurons.get(j).cacheWeights;
                layers.get(i).neurons.get(j).bias = layers.get(i).neurons.get(j).cacheBias;
            }
        }

    }

    public void train(int epochs, double learningRate, List<TrainingObject> to) {
        for(int i = 0; i < epochs; i++) {
            for(int j = 0; j < to.size(); j++) {
                forward(to.get(j).input);
                backPropagation(learningRate, to.get(j).expected);
            }
            System.out.printf("Epoch %d completed..\n", i);
        }
    }
}
