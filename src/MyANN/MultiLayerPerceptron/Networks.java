package MyANN.MultiLayerPerceptron;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.util.*;

/**
 *
 * @author Teofebano
 */
public class Networks {
    // Main Attributes
    public int numInputs;
    public int numOutputs;
    public int numHiddenLayer;
    public int numHiddenNeuron; 
    
    public Vector<NeuronLayer> networks;
    public Vector<Double> middleOutput;
    
    // CTOR
    public Networks(int numInputs, int numOutputs, int numHiddenLayer, int numHiddenNeuron){
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.numHiddenLayer = numHiddenLayer;
        this.numHiddenNeuron = numHiddenNeuron;
        
        networks = new Vector<>();
        middleOutput = new Vector<>();
    }
    
    // Create the "actual" net :v
    public void createNetworks(){
        if (this.numHiddenLayer > 0){
            // 1st layer
            networks.add(new NeuronLayer(this.numHiddenNeuron,this.numInputs));
            // next layer (if exist)
            for (int i=0;i<numHiddenLayer-1;i++){
                networks.add(new NeuronLayer(this.numHiddenNeuron,this.numHiddenNeuron));
            }
            // output layer
            networks.add(new NeuronLayer(this.numOutputs,this.numHiddenNeuron));
        }
        else{ // No hidden layer
            networks.add(new NeuronLayer(this.numOutputs,this.numInputs));
        }
    }
    
    // Find the number of all weights
    public int getNumberWeights(){
        int numWeight = 0;
        for (int i=0;i<numHiddenLayer+1;i++){ // + output
            // For each layer
            for (int j=0;j<networks.elementAt(i).numNeurons;j++){
                // For each neuron
                for (int k=0;k<networks.elementAt(i).neurons.elementAt(j).numInputs;k++){
                        numWeight++;
                }
            }
        }
        return numWeight;
    }
    
    // Find all the weights
    public Vector<Double> GetWeights(){
        Vector<Double> weight = new Vector<Double>();
        for (int i=0;i<numHiddenLayer+1;i++){
            // For each layer
            for (int j=0;j<networks.elementAt(i).numNeurons;j++){
                // For each neuron
                for (int k=0;k<networks.elementAt(i).neurons.elementAt(j).numInputs;k++){
                    weight.addElement(networks.elementAt(i).neurons.elementAt(j).weights.elementAt(k));
                }
            }
        }
        // NOTE !!!
        // 1st weight = weight for 1st neuron in 1st layer from 1st input
        // last weight = weight for last neuron in output layer from last neuron in previous hidden layer
        // I wish I can insert picture here :(
        return weight;
    }
    
    // Update weights
    public void PutWeights(Vector<Double> weight){
        int counter = 0;
        for (int i=0;i<numHiddenLayer+1;i++){
            // For each layer
            for (int j=0;j<networks.elementAt(i).numNeurons;j++){
                // For each neuron
                for (int k=0;k<networks.elementAt(i).neurons.elementAt(j).numInputs;k++){
                    networks.elementAt(i).neurons.elementAt(j).weights.setElementAt(weight.elementAt(counter), k);
                    counter++;
                }
            }
        }
    }
    
    // Calculate forward chaining
    public Vector<Double> ForwardChaining(Vector<Double> input){
        Vector<Double> inputs = new Vector<>();
        Vector<Double> outputs = new Vector<>();
        int counter; // counter for inputs
        // No. of inputs must be same as the default number of input
        if(input.size() != this.numInputs){
            return outputs; // empty Vector
        }
        // init. for input and middle output
        for (int i=0;i<input.size();i++){
            inputs.addElement(input.elementAt(i));
        }
        middleOutput.clear();

        for (int i=0;i<numHiddenLayer+1;i++){
            if (i>0){ // switching output as new input
                inputs.clear();
                for (int j=0;j<outputs.size();j++){
                    inputs.addElement(outputs.elementAt(j));
                    middleOutput.addElement(outputs.elementAt(j));
                }
            }
            outputs.clear();
            counter = 0;

            // For each layer
            for (int j=0;j<networks.elementAt(i).numNeurons;j++){
                double totAmount = 0;
                int numInputs = networks.elementAt(i).neurons.elementAt(j).numInputs;
                // For each neuron
                for (int k=0;k<numInputs-1;k++){
                    totAmount += inputs.elementAt(counter) * networks.elementAt(i).neurons.elementAt(j).weights.elementAt(k);
                    counter++;
                }
                totAmount += networks.elementAt(i).neurons.elementAt(j).weights.elementAt(numInputs-1); // bias
                // Storing
                outputs.addElement(Sigmoid(totAmount,1)); // Control = 1
                counter = 0;
            }
        }
        return outputs;
    }

    public double Sigmoid(double totalInput, double control){
        return (1/(1+ Math.exp(-totalInput/control)));
    }
}
