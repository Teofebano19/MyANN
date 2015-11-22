package SinglePerceptron;

/**
 *
 * @author Andrey
 */
public final class DeltaRuleBatch {
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
 
    DeltaRuleBatch(){
        currentepoch = 1;
        realoutput = 0.0;
        
        while(currentepoch <= maxepoch || isConvergent){
            for(int i = 0; i < neuron.numinstances; i++){
                CountRealOutput(i);
                output = realoutput;
                realoutput = 0.0;
                targetminoutput = neuron.targets.elementAt(i) - output;
                for(int j = 0; j < neuron.numinputs; j++){
                    deltaweight = learningrate * targetminoutput * neuron.data.elementAt(i).elementAt(j); 
                    updatedweight = neuron.weights.elementAt(j) + deltaweight;
                    neuron.weights.setElementAt(updatedweight,j);
                }
            }
            CountError();
            if(error < errortreshold){
                isConvergent = true;
            }
            currentepoch++;
        }
        
        System.out.println("Epoch sekarang: "+currentepoch);
    }
    
    public void CountRealOutput(int d){
        for(int i = 0; i < neuron.numinputs; i++){    
            realoutput += neuron.data.elementAt(d).elementAt(i) * neuron.weights.elementAt(i);
        }
    }
    
    public void CountError(){
        for(int i = 0; i < neuron.numinstances; i++){
            for(int j = 0; j < neuron.numinputs; j++){
                realoutput += neuron.data.elementAt(i).elementAt(j) * neuron.weights.elementAt(j);
            }
            output = realoutput;
            realoutput = 0.0;
            targetminoutput = neuron.targets.elementAt(i) - output;
            //count error
            error += targetminoutput*targetminoutput;
        }
        error = error/2;
    }    
}
