package MyANN.SingleLayerPerceptron;



import MyANN.Helper;
import java.util.*;
import weka.core.Instances;
        
/**
 *
 * @author Andrey
 */
public final class DeltaRuleIncremental {
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
    SingleLayerPerceptron neuron = new SingleLayerPerceptron() {

        @Override
        public void buildClassifier(Instances i) throws Exception {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    };
    Helper helper = new Helper();
 
    public DeltaRuleIncremental(){
        currentepoch = 1;
        realoutput = 0.0;
        isConvergent = false;
        error = 1000000.0;
    }
    
    public void run(){
        while(currentepoch <= maxepoch && !isConvergent){
            System.out.println("Epoch: "+currentepoch);
            for(int i = 0; i < neuron.numinstances; i++){
                System.out.println("Instance ke-"+i );
                CountRealOutput(i);
                output = realoutput;
                realoutput = 0.0;
                targetminoutput = neuron.targets.elementAt(i) - output;
                for(int j = 0; j < neuron.numinputs; j++){
                    deltaweight = learningrate * targetminoutput * 
                                  neuron.data.elementAt(i).elementAt(j); 
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
        }
    }
    
    public void CountRealOutput(int d){
        for(int i = 0; i < neuron.numinputs; i++){ 
            realoutput += neuron.data.elementAt(d).elementAt(i) * neuron.weights.elementAt(i);
            System.out.println("x ke-"+i+" = "+neuron.data.elementAt(d).elementAt(i));
            System.out.println("weight ke-"+i+" = "+neuron.weights.elementAt(i));
        }
        System.out.println("realoutput di fungsi= "+realoutput);
        System.out.println("");
    }
    
    public void CountError(){
        for(int i = 0; i < neuron.numinstances; i++){
            for(int j = 0; j < neuron.numinputs; j++){
                realoutput += neuron.data.elementAt(i).elementAt(j) * neuron.weights.elementAt(j);
            }
            realoutput = 0.0;
            targetminoutput = neuron.targets.elementAt(i) - output;
            error += targetminoutput*targetminoutput;
        }
        error = error/2;
    }
}