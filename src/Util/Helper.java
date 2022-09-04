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
}
