package SinglePerceptron;

import java.util.*;
        
/**
 *
 * @author Andrey
 */
public class PerceptronTrainingRule {
    public static Double learningrate = 0.1;
    public static int maxepoch = 10;
    public static Double errortreshold = 0.1;
    public SingleNeuronLayer neuronlayer;
    public boolean isConvergent;
    private int currentepoch;
    
    PerceptronTrainingRule(){
        while(currentepoch <= maxepoch || isConvergent){
            
        }
    }
}


