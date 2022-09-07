package Network;

import Training.TrainingObject;
import Util.Helper;

import java.util.*;

public class NeuralNetwork {

    public double error = 0;

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

    public void backPropagation(double learningRate, double[] expected) {
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
            cost_activatedVal = Helper.qcfDerivative(expected[i], neuron.value);
            activatedVal_weightedVal = Helper.sigmoidDerivative(neuron.weightedValue);
            double delta = cost_activatedVal * activatedVal_weightedVal;
            totalDelta += delta;
            error += Helper.quadraticCostFunction(expected[i], neuron.value);
//            error += 0.5 * (Math.pow((expected[i] - neuron.value), 2));
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
        System.out.println("=====================");
        System.out.println("Starting training...");
        double min = Integer.MAX_VALUE;

        for(int i = 0; i < epochs; i++) {
            //double avgError = 0;
            for(int j = 0; j < to.size(); j++) {
                forward(to.get(j).input);
                backPropagation(learningRate, to.get(j).expected);
                //avgError += error;
                if(j % 1000 == 0) {
                    error = error / layers.get(layers.size()-1).neurons.size();
                    //System.out.println("The error at i " + j + " is " + error);
//                    min = Math.min(min, error) > 1 ? Math.min(min, error) : min;
                    if(i > 0) min = Math.min(error, min);
//                    if(error < 5 && error > 1) {
//                        System.out.println("Broke out early");
//                        System.out.println("Error min was " + error);
//                        return;
//                    }
                    error = 0;
                }
            }
//            System.out.println("The avg error is: " + avgError/1000);

//            Collections.shuffle(to);
//            to = Helper.shuffle(to);
//            System.out.println("Epoch finished: " + i);
            if(i % 100000 == 0 && i > 0) System.out.printf("Epoch %d completed..\n", i);
        }
        System.out.println("Training completed!");
        System.out.println("=====================");
        System.out.println("the minimum error found was: " + min);
    }

    public double test(List<TrainingObject> to) {
        double correct = 0;
        for(int i = 0; i < to.size(); i ++) {
            forward(to.get(i).input);
            double max = 0;
            int index = 0;
            for(int j = 0; j < layers.get(layers.size()-1).neurons.size(); j++) {
                double val = layers.get(layers.size()-1).neurons.get(j).value;
                if(val > max) {
                    max = val;
                    index = j;
                }
            }
            double[] actual = Helper.getLabelArray(index, 10);
            if(Arrays.equals(actual, to.get(i).expected)) correct++;
        }
        System.out.println(to.size());
        return correct / to.size();
    }

    public double testXOR(List<TrainingObject> to) {
        int count = 0;
        for(int i = 0; i < 100; i++) {
            for(int j = 0; j < to.size(); j++) {
                forward(to.get(j).input);
                double val = layers.get(layers.size()-1).neurons.get(0).value;
                if(Arrays.equals(to.get(j).expected, new double[]{Math.round(val)})) count++;
            }
        }
        return count / to.size();
    }
}
