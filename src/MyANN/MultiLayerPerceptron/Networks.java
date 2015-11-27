package MyANN.MultiLayerPerceptron;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.util.*;
import MyANN.Helper;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author Teofebano
 */
public class Networks extends Classifier{
    // Main Attributes
    public int numInputs;
    public int numOutputs;
    public int numHiddenLayer;
    public int numHiddenNeuron;
    public double RATE;
    
    public Vector<NeuronLayer> networks;
    public Vector<Double> middleOutput;
    
    // CTOR
    public Networks(int numInputs, int numOutputs, int numHiddenLayer, int numHiddenNeuron, double RATE){
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.numHiddenLayer = numHiddenLayer;
        this.numHiddenNeuron = numHiddenNeuron;
        this.RATE = RATE;
        
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
        Helper helper = new Helper();
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
                outputs.addElement(helper.SigmoidActivationFunction(totAmount,1)); // Control = 1
                counter = 0;
            }
        }
        return outputs;
    }
    
    public Vector<Double> calculateError(Vector<Double> input, Vector<Double> targetOutput){
        Vector<Double> actualOutput = new Vector<Double>();
        Vector<Double> error = new Vector<Double>();
        for (int i=0;i<actualOutput.size();i++){
            error.addElement(new Double(actualOutput.elementAt(i)*(1-actualOutput.elementAt(i))*(targetOutput.elementAt(i)-actualOutput.elementAt(i))));
        }
        return error;
    }
    
    public void backPropagate(Vector<Double> input, Vector<Double> targetOutput){
        Vector<Double> errorHidden = new Vector<Double>();
        Vector<Double> error = new Vector<Double>();
        Vector<Double> oldWeight = this.GetWeights();
        
        // Calculating hidden error
        for (int i=this.numHiddenLayer;i>0;i--){
            error.clear();
            if (i==this.numHiddenLayer){
                error = calculateError(input, targetOutput);
            }
            else{
                for (int j=errorHidden.size()-numHiddenNeuron;j<errorHidden.size();j++){
                        error.addElement(errorHidden.elementAt(j));
                }
            }
            for (int j=numHiddenNeuron;j>0;j--){
                Vector<Double> sumError = new Vector<Double>();
                for (int k=error.size();k>0;k--){
                    if (i==numHiddenLayer){
                        sumError.addElement(new Double(error.elementAt(k-1)*oldWeight.elementAt(
                            ((numInputs+1)*numHiddenNeuron+(numHiddenNeuron+1)*numHiddenNeuron*(numHiddenLayer-1)+(numHiddenNeuron+1)*numOutputs)-2-((numHiddenNeuron+1)*(numOutputs-k))-(numHiddenNeuron-j)
                        )));
                    }
                    else{
                        sumError.addElement(new Double(error.elementAt(k-1)*oldWeight.elementAt(
                                ((numInputs+1)*numHiddenNeuron+(numHiddenNeuron+1)*numHiddenNeuron*i)-2-((numHiddenNeuron+1)*(numHiddenNeuron-k))-(numHiddenNeuron-j)
                        )));
                    }
                }
                errorHidden.addElement(total(sumError)*this.middleOutput.elementAt((i*j)-1)*(1-this.middleOutput.elementAt((i*j)-1)));
            }
        }
        
        // Changing the Weight
        Vector<Double> newWeight = new Vector<Double>();
        Vector<Double> inputs = new Vector<Double>();
        int counterWeight = 0;
        for (int i=0;i<this.numHiddenLayer+1;i++){ // including Output
            inputs.clear();
            if (i==0){
                for (int j=0;j<input.size();j++){
                    inputs.addElement(input.elementAt(j));
                }
            }
            else{
                for (int j=0;j<this.middleOutput.size();j++){
                    inputs.addElement(this.middleOutput.elementAt(j));
                }
            }
            for (int j=0;j<this.networks.elementAt(i).numNeurons;j++){ 
                for (int k=0;k<this.networks.elementAt(i).neurons.elementAt(j).numInputs-1;k++){ // excluding bias
                    if (i==this.numHiddenLayer){
                            newWeight.addElement(oldWeight.elementAt(counterWeight)+inputs.elementAt(k)*this.RATE*error.elementAt(j));
                    }
                    else{
                            newWeight.addElement(oldWeight.elementAt(counterWeight)+inputs.elementAt(k)*this.RATE*errorHidden.elementAt(errorHidden.size()-this.numHiddenNeuron*i-j-1));
                    }
                    counterWeight++;
                }
                // bias
                if (i==this.numHiddenLayer){
                    newWeight.addElement(oldWeight.elementAt(counterWeight)+RATE*error.elementAt(j));
                }
                else{
                    newWeight.addElement(oldWeight.elementAt(counterWeight)+RATE*errorHidden.elementAt(errorHidden.size()-this.numHiddenNeuron*i-j-1));
                }
                counterWeight++;
            }
        }
        PutWeights(newWeight);
    }
    
    private synchronized double total(Vector<Double> total){
        double x = 0;
        for (int i=0;i<total.size();i++){
                x+=total.elementAt(i);
        }
        return x;
    }
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
