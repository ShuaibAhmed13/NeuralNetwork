import Network.Layer;
import Network.NeuralNetwork;
import Network.Neuron;
import Training.TrainingObject;
import Util.Helper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
        List<TrainingObject> to = new ArrayList<>();
        to.add(new TrainingObject(new double[]{0, 0}, 0));
        to.add(new TrainingObject(new double[]{0, 1}, 1));
        to.add(new TrainingObject(new double[]{1, 0}, 1));
        to.add(new TrainingObject(new double[]{1, 1}, 0));
        nn.train(350000, 0.1, to);
        System.out.println("After training for input {0,0}");
        nn.forward(new double[]{0, 0});
        for (int i = 0; i < nn.layers.get(nn.layers.size() - 1).neurons.size(); i++) {
            System.out.println(nn.layers.get(nn.layers.size() - 1).neurons.get(i).value);
        }
        nn.forward(new double[]{1, 0});
        for (int i = 0; i < nn.layers.get(nn.layers.size() - 1).neurons.size(); i++) {
            System.out.println(nn.layers.get(nn.layers.size() - 1).neurons.get(i).value);
        }nn.forward(new double[]{0, 1});
        for (int i = 0; i < nn.layers.get(nn.layers.size() - 1).neurons.size(); i++) {
            System.out.println(nn.layers.get(nn.layers.size() - 1).neurons.get(i).value);
        }

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
