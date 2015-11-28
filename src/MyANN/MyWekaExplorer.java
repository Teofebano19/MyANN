package MyANN;

import MyANN.neurons.SigmoidNeuron;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MyWekaExplorer {

    private Classifier classifier;

    private Instances trainingData;

    private Instances testData;

    public void readTrainingDataFromArff(String filename) throws Exception{
        trainingData = DataSource.read(filename);
        trainingData.setClassIndex(trainingData.numAttributes()-1);
    }

    public void readTestDataFromArff(String filename) throws Exception{
        testData = DataSource.read(filename);
        testData.setClassIndex(trainingData.numAttributes()-1);
    }

    public void removeAttribute(int[] idx) throws Exception{
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(idx);
        remove.setInvertSelection(true);
        remove.setInputFormat(trainingData);
        trainingData =  Filter.useFilter(trainingData,remove);
    }

    public void buildClassifier(Classifier classifier) throws Exception{
        this.classifier = classifier;
        classifier.buildClassifier(trainingData);

    }

    public void testModel() throws Exception{
        Evaluation eval = new Evaluation(testData);
        eval.evaluateModel(classifier, testData);
        System.out.println(eval.toSummaryString("Results",false));
    }

    public void crossValidation() throws Exception{
        Evaluation eval = new Evaluation(trainingData);
        eval.crossValidateModel(classifier, trainingData, 10, new Random(1));
        System.out.println(eval.toSummaryString("Results", false));
    }



    public void saveModel(String filename) throws Exception{
        SerializationHelper.write(filename, classifier);
    }

    public void loadModel(String filename) throws Exception{
        classifier = (Classifier) SerializationHelper.read(filename);
    }

    public void classify(String filename) throws Exception{
        Instances unLabeledData = DataSource.read(filename);
        unLabeledData.setClassIndex(unLabeledData.numAttributes()-1);
        Instances LabeledData = new Instances(unLabeledData);

        for(int i=0; i < unLabeledData.numInstances();++i){
            double clsLabel = classifier.classifyInstance(unLabeledData.instance(i));
            LabeledData.instance(i).setClassValue(clsLabel);
        }
        System.out.println(LabeledData.toString());
    }

    public static void main(String argv[]){
        MyWekaExplorer wekaInterface = new MyWekaExplorer();
        String[] data = new String[]{
//                "data/weather.numeric.arff"};
                "data/iris.arff"};
        Classifier classifier = new MyANN(new ArrayList<>(), new SigmoidNeuron());
        int numData = data.length;
        for (int i = 0; i < numData; ++i) {
            System.out.printf("===================================================\n");
            System.out.printf("for training data [%s]\n", data[i]);
            try {
                wekaInterface.readTrainingDataFromArff(data[i]);
                wekaInterface.buildClassifier(classifier);
                wekaInterface.crossValidation();
            } catch (Exception ex) {
                System.out.println("Error was occured: " + ex.getMessage());
                ex.printStackTrace();
            }
            System.out.printf("===================================================\n\n");
        }
    }

}
