/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MultiLayerPerceptron;

import java.util.*;
import MultiLayerPerceptron.Neuron;

/**
 *
 * @author Teofebano
 */
public class NeuronLayer {
    // Attributes
    public int numNeurons;
    public Vector<Neuron> neurons;
    
    // CTOR
    NeuronLayer(int numNeurons, int numInputperNeurons){
        this.numNeurons = numNeurons;
        neurons = new Vector<>();
        for (int i=0;i<numNeurons;i++){
            Neuron neuron = new Neuron(numInputperNeurons);
            neurons.addElement(neuron);
        }
    }
}
