package Util;

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
}
