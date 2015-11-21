/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package SinglePerceptron;

/**
 *
 * @author Andrey
 */
public class Helper {
    public Double SigmoidActivationFunction(Double d){
        return 0.0;
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
