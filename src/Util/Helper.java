package Util;

import Training.TrainingObject;

import java.util.List;
import java.util.Random;

public class Helper {

    static Random random = new Random(123);
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

    public static double reLU(double x) {
        return Math.max(0, x);
    }

    public static double reLUDerivative(double x) {
        return x < 0 ? 0 : 1;
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

    public static double[] getLabelArray(int val, int size) {
        double[] array = new double[size];
        array[val] = 1;
        return array;
    }
}
