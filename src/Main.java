import Network.Layer;
import Network.NeuralNetwork;
import Network.Neuron;
import Training.Image;
import Training.TrainingObject;
import Util.Helper;
import Util.MNIST_Reader;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork();
        nn.addLayer(new double[]{1, 0});
        nn.addLayer(2, 4);
        nn.addLayer(4, 1);
        System.out.println("Before training for input {0,0}");
        nn.forward(new double[]{0, 0});
        for (int i = 0; i < nn.layers.get(nn.layers.size() - 1).neurons.size(); i++) {
            System.out.println(nn.layers.get(nn.layers.size() - 1).neurons.get(i).value);
        }
        System.out.println("Result for input {1,0} before training:");
        nn.forward(new double[]{1, 0});
        for (int i = 0; i < nn.layers.get(nn.layers.size() - 1).neurons.size(); i++) {
            System.out.println(nn.layers.get(nn.layers.size() - 1).neurons.get(i).value);
        }
        System.out.println();
        List<TrainingObject> to = new ArrayList<>();
        to.add(new TrainingObject(new double[]{0, 0}, new double[]{0}));
        to.add(new TrainingObject(new double[]{0, 1}, new double[]{1}));
        to.add(new TrainingObject(new double[]{1, 0}, new double[]{1}));
        to.add(new TrainingObject(new double[]{1, 1}, new double[]{0}));
        nn.train(3500000, 0.1, to);
        System.out.println("\nResult for input {0,0} after training:");
        nn.forward(new double[]{0, 0});
        for (int i = 0; i < nn.layers.get(nn.layers.size() - 1).neurons.size(); i++) {
            System.out.println(nn.layers.get(nn.layers.size() - 1).neurons.get(i).value);
        }
        System.out.println("Before training for input {1,0}");
        nn.forward(new double[]{1, 0});
        for (int i = 0; i < nn.layers.get(nn.layers.size() - 1).neurons.size(); i++) {
            System.out.println(nn.layers.get(nn.layers.size() - 1).neurons.get(i).value);
        }
        System.out.println();
        for(int i = 0; i < 396; i++) {
            to.add(to.get(new Random().nextInt(4)));
        }

        System.out.println("The test size is: " + to.size());
        System.out.println("Neural Network accuracy in performing XOR operations: ");
        System.out.println(nn.testXOR(to) + "%");







//        NeuralNetwork nn = new NeuralNetwork();
//        nn.addLayer(new double[784]);
//        nn.addLayer(784, 100);
//        nn.addLayer(100, 10);
//        List<Image> images = MNIST_Reader.readMNIST("MNIST_DATA/mnist_train.csv");
//        List<TrainingObject> to = new ArrayList<>();
//        for(int i = 0; i < images.size(); i++) {
//            Image image = images.get(i);
//            to.add(new TrainingObject(image.imageMatrix, Helper.getLabelArray(image.label, 10)));
//        }
//        System.out.println("Input is number 5");
//        nn.forward(to.get(0).input);
//        double max = 0;
//        int index = 0;
//        for(int i = 0; i < nn.layers.get(nn.layers.size()-1).neurons.size(); i++) {
//            double val = nn.layers.get(nn.layers.size()-1).neurons.get(i).value;
//            System.out.println("Neuron no " + i + ": " + val);
//            if(val > max) {
//                max = val;
//                index = i;
//            }
//        }
//        System.out.println("the answer index is: " + index);
//        nn.train(10, .01, to);
//        System.out.println("Input is number 5");
//        nn.forward(to.get(0).input);
//        max = 0;
//        index = 0;
//        for(int i = 0; i < nn.layers.get(nn.layers.size()-1).neurons.size(); i++) {
//            double val = nn.layers.get(nn.layers.size()-1).neurons.get(i).value;
//            System.out.println("Neuron no " + i + ": " + val);
//            if(val > max) {
//                max = val;
//                index = i;
//            }
//        }
//        System.out.println(Arrays.toString(to.get(0).expected));
//        System.out.println("The answer index is: " + index);
//
//        System.out.println("Input is number 0");
//        nn.forward(to.get(1).input);
//        max = 0;
//        index = 0;
//        for(int i = 0; i < nn.layers.get(nn.layers.size()-1).neurons.size(); i++) {
//            double val = nn.layers.get(nn.layers.size()-1).neurons.get(i).value;
//            System.out.println("Neuron no " + i + ": " + val);
//            if(val > max) {
//                max = val;
//                index = i;
//            }
//        }
//        System.out.println(Arrays.toString(to.get(1).expected));
//        System.out.println("The answer index is: " + index);
//
//        System.out.println("Input is number 4");
//        nn.forward(to.get(2).input);
//        max = 0;
//        index = 0;
//        for(int i = 0; i < nn.layers.get(nn.layers.size()-1).neurons.size(); i++) {
//            double val = nn.layers.get(nn.layers.size()-1).neurons.get(i).value;
//            System.out.println("Neuron no " + i + ": " + val);
//            if(val > max) {
//                max = val;
//                index = i;
//            }
//        }
//        System.out.println(Arrays.toString(to.get(2).expected));
//        System.out.println("The answer index is: " + index);
//
//
//        List<Image> testImages = MNIST_Reader.readMNIST("MNIST_DATA/mnist_test.csv");
//        List<TrainingObject> testList = new ArrayList<>();
//        for(int i = 0; i < testImages.size(); i++) {
//            testList.add(new TrainingObject(testImages.get(i).imageMatrix,
//                    Helper.getLabelArray(testImages.get(i).label, 10)));
//        }
//        System.out.println("the accuracy percentage is: " + nn.test(testList));




//        System.out.println(images.size());
//        System.out.println(images.get(5).imageMatrix);
//        System.out.println(images.get(5).label);
//        for(int i = 0; i < nn.layers.size(); i++) {
//            System.out.println("Layer no: " + i);
//            for(Neuron n: nn.layers.get(i).neurons) {
//                System.out.println(Arrays.toString(n.weights));
//                System.out.println("bias: " + n.bias);
//                System.out.println(n.value + "\n");
//            }
//        }


    }
}
