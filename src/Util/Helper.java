package Util;

import Training.TrainingObject;

import java.util.List;
import java.util.Random;

public class Helper {

    static Random random = new Random(0);
    public static double getRandomDouble() {
        return random.nextGaussian();
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    public static double sigmoidDerivative(double x) {
        double sigmoid = sigmoid(x);
        return sigmoid * (1-sigmoid);
    }

    public static double quadraticCostFunction(double expected, double actual) {
        return 0.5 * Math.pow((expected - actual), 2);
    }

    public static double qcfDerivative(double expected, double actual) {
        return actual-expected;
    }

    public static List<TrainingObject> shuffle(List<TrainingObject> to) {
        Random random = new Random();
        for(int i = to.size()-1; i > 0; i--) {
            int randomVal = random.nextInt(i);
            TrainingObject toTemp = to.get(randomVal);
            to.set(randomVal, to.get(i));
            to.set(i, toTemp);
        }
        return to;
    }
}
