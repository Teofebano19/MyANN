package MyANN;

import java.util.*;

public class Neuron {
    // Attributes
    public int numInputs;
    public Vector<Double> weights;
    public double bias = 1;
    public double biasweight;
    public Helper helper;
    public double output;
    
    // CTOR
    public Neuron(int numInputs){
        weights = new Vector<>();
        Random ran = new Random();
        this.numInputs = numInputs+1; // including bias
        for (int i=0;i<this.numInputs;i++){
            weights.addElement(-1+(2*ran.nextDouble()));
        }
    }
    
}
