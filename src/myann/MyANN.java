package myann;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import SinglePerceptron.PerceptronTrainingRule;
import SinglePerceptron.DeltaRuleBatch;
import SinglePerceptron.SingleNeuron;
import SinglePerceptron.Helper;

/**
 *
 * @author Teofebano
 */
public class MyANN {
    // Attributes
    private static final String SOURCE = "data/iris.arff";
    private static final int NUMBER_FOLD = 10;
    private static final int PERCENTAGE = 66;
    public static Instances data;
    
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        //MyANN main = new MyANN();
        //main.run();
        
        DeltaRuleBatch test = new DeltaRuleBatch();
        test.run();
        
    }
    
    private void run(){
        loadFile(SOURCE);
    }
    
    // Load file
    public static void loadFile(String source){
        try {
            data = ConverterUtils.DataSource.read(source);
            if (data.classIndex() == -1){
                data.setClassIndex(data.numAttributes()-1);
            }
        } catch (Exception ex) {
            Logger.getLogger(MyANN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // percentage split
    public static void learnPercentage(Instances trainingData, Classifier classifier){
        try {
            // Build
            Classifier cls = classifier;
            int trainSize = trainingData.numInstances() * PERCENTAGE;
            int testSize = trainingData.numInstances();
            Instances train = new Instances(trainingData, 0, trainSize);
            Instances test = new Instances(trainingData, trainSize, testSize);
            cls.buildClassifier(train);
            
            // Eval
            Evaluation eval = new Evaluation(trainingData);
            eval.evaluateModel(classifier, test);
            
            // Print
            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(MyANN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // full
    public static void learnFull(Instances trainingData, Classifier classifier){
        try {
            // Build
            Classifier cls = classifier;
            int trainSize = trainingData.numInstances();
            Instances train = new Instances(trainingData, 0, trainSize);
            Instances test = new Instances(trainingData, 0, trainSize);
            cls.buildClassifier(train);
            
            // Eval
            Evaluation eval = new Evaluation(trainingData);
            eval.evaluateModel(classifier, test);
            
            // Print
            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(MyANN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // 10-fold
    public static void learn10fold(Instances trainingData, Classifier classifier){
        try {
            // Build and Eval
            Evaluation eval = new Evaluation(trainingData);
            eval.crossValidateModel(classifier, trainingData, NUMBER_FOLD, new Debug.Random(1));
            
            // Print
            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(MyANN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
