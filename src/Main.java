import Network.Layer;
import Network.NeuralNetwork;
import Network.Neuron;

public class Main {

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork();
        nn.addLayer(new double[]{1,0});
        nn.addLayer(2, 4);
        nn.addLayer(4, 1);
        nn.forward();
        for(int i = 0; i < nn.layers.size(); i++) {
            System.out.println("Layer no: " + i);
            for(Neuron n: nn.layers.get(i).neurons) {
                System.out.println(n.value + "\n");
            }
        }

    }
}
