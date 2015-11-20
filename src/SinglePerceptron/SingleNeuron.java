/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package SinglePerceptron;

import java.util.Random;
import java.util.Vector;

/**
 *
 * @author Andrey
 */
public class SingleNeuron {
    // Attributes
    public int numInputs;
    public Vector<Double> weights;
    
    // CTOR
    SingleNeuron(int numInputs){
        weights = new Vector<>();
        Random ran = new Random();
        this.numInputs = numInputs+1; // including bias
        for (int i=0;i<this.numInputs;i++){
            weights.addElement(-1+(2*ran.nextDouble()));
        }
    }
}
