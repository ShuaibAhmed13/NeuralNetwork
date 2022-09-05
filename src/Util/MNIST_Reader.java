package Util;

import Training.Image;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class MNIST_Reader {

    public static List<Image> readMNIST(String filePath) {
        List<Image> images = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(filePath)));
            String line = br.readLine();
            while(line != null) {
                String[] values = line.split(",");
                double[] matrix = new double[values.length];
                for(int i = 1; i < values.length; i++) {
                    matrix[i] = (double) Integer.parseInt(values[i]);
                }
                images.add(new Image(matrix, Integer.parseInt(values[0])));
                line = br.readLine();
            }
        } catch (Exception e) {
            System.out.println("This file could not be found!");
        }
        return images;
    }
}
