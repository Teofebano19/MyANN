package MyANN;



/**
 *
 * @author Andrey
 */
public class Helper {
    public Double SigmoidActivationFunction(double totalInput, double control){
        return (1/(1+ Math.exp(-totalInput/control)));
    }

    public Double SignActivationFunction(Double d){
        if(d >= 0){
            return 1.0;
        }
        else{
            return -1.0;
        }       
    }
    
    public Double StepActivationFunction(Double d){
        if(d >= 0){
            return 1.0;
        }
        else{
            return 0.0;
        }    
    }
}
