/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package SinglePerceptron;

import SinglePerceptron.SingleNeuron;
import java.util.Vector;

/**
 *
 * @author Andrey
 */
public class SingleNeuronLayer {
    // Attributes
    public int numNeurons;
    public Vector<SingleNeuron> neurons;
    
    // CTOR
    SingleNeuronLayer(int numNeurons, int numInputperNeurons){
        this.numNeurons = numNeurons;
        neurons = new Vector<>();
        for (int i=0;i<numNeurons;i++){
            SingleNeuron neuron = new SingleNeuron(numInputperNeurons);
            neurons.addElement(neuron);
        }
    }
}
