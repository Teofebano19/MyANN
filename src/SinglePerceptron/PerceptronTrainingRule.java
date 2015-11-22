package SinglePerceptron;

import java.util.*;
        
/**
 *
 * @author Andrey
 */
public final class PerceptronTrainingRule {
    public static Double learningrate = 0.1;
    public static int maxepoch = 10;
    public static Double errortreshold = 0.01;
    public boolean isConvergent;
    private int currentepoch;   
    public Double realoutput;
    public Double output;
    public Double error;
    public Double targetminoutput;
    public Double deltaweight;
    public Double updatedweight;
    SingleNeuron neuron = new SingleNeuron();
    Helper helper = new Helper();
 
    public PerceptronTrainingRule(){
        currentepoch = 1;
        realoutput = 0.0;
        
        while(currentepoch <= maxepoch || isConvergent){
            for(int i = 0; i < neuron.numinstances; i++){
                CountRealOutput(i);
                output = helper.SignActivationFunction(realoutput);
                System.out.println("Realoutput di epoch = "+realoutput);
                System.out.println("");
                realoutput = 0.0;
                targetminoutput = neuron.targets.elementAt(i) - output;
                for(int j = 0; j < neuron.numinputs; j++){
                    deltaweight = learningrate * targetminoutput * neuron.data.elementAt(i).elementAt(j); 
                    updatedweight = neuron.weights.elementAt(j) + deltaweight;
                    neuron.weights.setElementAt(updatedweight,j);
                }
            }
            error = 0.0;
            CountError();
            if(error <= errortreshold){
                isConvergent = true;
            }
            currentepoch++;
//            System.out.println("Epoch: "+currentepoch);
        }
    }
    
    public void CountRealOutput(int d){
        for(int i = 0; i < neuron.numinputs; i++){ 
            realoutput += neuron.data.elementAt(d).elementAt(i) * neuron.weights.elementAt(i);
//            System.out.println("x ke-"+i+" = "+neuron.data.elementAt(d).elementAt(i));
//            System.out.println("weight ke-"+i+" = "+neuron.weights.elementAt(i));
//            
//            System.out.println("realoutput di fungsi= "+realoutput);
            System.out.println(neuron.data.elementAt(0).elementAt(i));
            System.out.println(neuron.data.elementAt(1).elementAt(i));
            System.out.println(neuron.data.elementAt(2).elementAt(i));
        }
    }
    
    public void CountError(){
        for(int i = 0; i < neuron.numinstances; i++){
            for(int j = 0; j < neuron.numinputs; j++){
                realoutput += neuron.data.elementAt(i).elementAt(j) * neuron.weights.elementAt(j);
            }
//            System.out.println("Real output = "+realoutput);
            output = helper.SignActivationFunction(realoutput);
//            System.out.println("Output = "+output);
//            System.out.println("target = "+neuron.targets.elementAt(i));
            realoutput = 0.0;
            targetminoutput = neuron.targets.elementAt(i) - output;
//            System.out.println("target-output = "+targetminoutput);
//            System.out.println("");
            //count error
            error += targetminoutput*targetminoutput;
        }
        error = error/2;
    }
}


